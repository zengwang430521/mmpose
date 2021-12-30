#include "kernels.cuh"

torch::Tensor similar_cuda_forward(
        const torch::Tensor &x_ori,
        const torch::Tensor &x_loc,
        const int kH, const int kW
) {
    TypeCheck(x_ori);
    TypeCheck(x_loc);
    const int batch = x_ori.size(0);
    const int channels = x_ori.size(1);
    const int height = x_ori.size(2);
    const int width = x_ori.size(3);

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;
    const int per_output = height * width * patch;
    auto output = torch::empty({batch, height, width, patch}, x_ori.options());

    int start_inp = 0, start_out = 0;
    for (int i = 0; i < batch; ++i) {
        f_cc2k<float, double>(
                at::cuda::getCurrentCUDAStream(),
                x_ori.data_ptr<float>() + start_inp,
                x_loc.data_ptr<float>() + start_inp,
                kH, kW, rH, rW,
                patch, channels, height, width,
                per_channel,
                output.data_ptr<float>() + start_out
        );
        start_inp += per_input;
        start_out += per_output;
    }

    return output;
}

//////////////////////////////////////////////////////////////

torch::Tensor similar_cuda_backward(
        const torch::Tensor &x,
        const torch::Tensor &grad_out,
        const int kH, const int kW,
        const bool is_ori
) {
    TypeCheck(x);
    const int batch = x.size(0);
    const int channels = x.size(1);
    const int height = x.size(2);
    const int width = x.size(3);

    const int rH = kH >> 1;
    const int rW = kW >> 1;
    const int patch = kH * kW;
    const int per_channel = height * width;
    const int per_input = per_channel * channels;

    auto grad_inp = torch::empty({batch, channels, height, width}, x.options());

    int start_inp = 0;
    for (int i = 0; i < batch; ++i) {
        auto grad_out_row = grad_out.select(0, i);
        if (is_ori) {
            f_ck2c_ori<float, double>(
                    at::cuda::getCurrentCUDAStream(),
                    x.data_ptr<float>() + start_inp,
                    grad_out_row.data_ptr<float>(),
                    kH, kW, rH, rW,
                    patch, channels,
                    height, width,
                    per_channel, per_input,
                    grad_inp.data_ptr<float>() + start_inp
            );
        } else {
            f_ck2c_loc<float, double>(
                    at::cuda::getCurrentCUDAStream(),
                    x.data_ptr<float>() + start_inp,
                    grad_out_row.data_ptr<float>(),
                    kH, kW, rH, rW,
                    patch, channels,
                    height, width,
                    per_channel, per_input,
                    grad_inp.data_ptr<float>() + start_inp
            );
        }
        start_inp += per_input;
    }
    return grad_inp;
}



////////////////////////////////////////////////////////////////////

torch::Tensor distance_cuda_forward(
        const torch::Tensor &query,
        const torch::Tensor &key,
        const torch::Tensor &idx
) {
    TypeCheck(query);
    TypeCheck(key);
    const int batch = query.size(0);
    const int Nquery = query.size(1);
    const int channels = query.size(2);
    const int Nkey = key.size(1);
    const int kernel = idx.size(2);
    auto output = torch::empty({batch, Nquery, kernel}, query.options());

    f_mydist<float, double>(
            at::cuda::getCurrentCUDAStream(),
            query.data_ptr<float>(),
            key.data_ptr<float>(),
            idx.data_ptr<int>(),
            batch, Nquery, Nkey, kernel, channels,
            output.data_ptr<float>()
    );
    return output;
}


////////////////////////////////////////////////////////////////////

// from query, key to attn
torch::Tensor query_key2attn_cuda(
        const torch::Tensor &query,
        const torch::Tensor &key,
        const torch::Tensor &idx
) {
    TypeCheck(query);
    TypeCheck(key);
    const int batch = query.size(0);
    const int Nquery = query.size(1);
    const int channels = query.size(2);
    const int Nkey = key.size(1);
    const int kernel = idx.size(2);
    auto output = torch::empty({batch, Nquery, kernel}, query.options());

    f_qk2attn<float, double>(
            at::cuda::getCurrentCUDAStream(),
            query.data_ptr<float>(),
            key.data_ptr<float>(),
            idx.data_ptr<int>(),
            batch, Nquery, Nkey, kernel, channels,
            output.data_ptr<float>()
    );
    return output;
}


// from attn, key to query

torch::Tensor attn_key2query_cuda(
        const torch::Tensor &attn,
        const torch::Tensor &key,
        const torch::Tensor &idx
) {
    TypeCheck(attn);
    TypeCheck(key);

    const int batch = key.size(0);
    const int Nkey = key.size(1);
    const int channels = key.size(2);
    const int Nquery = idx.size(1);
    const int kernel = idx.size(2);

    auto output = torch::empty({batch, Nquery, channels}, attn.options());

    f_attn_key2query<float, double>(
            at::cuda::getCurrentCUDAStream(),
            attn.data_ptr<float>(),
            key.data_ptr<float>(),
            idx.data_ptr<int>(),
            batch, Nquery, Nkey, kernel, channels,
            output.data_ptr<float>()
    );
    return output;
}


//from attn, query to key

torch::Tensor attn_query2key_cuda(
        const torch::Tensor &attn,
        const torch::Tensor &query,
        const torch::Tensor &idx,
        const int Nkey

) {
    TypeCheck(attn);
    TypeCheck(query);

    const int batch = query.size(0);
    const int Nquery = query.size(1);
    const int channels = query.size(2);
    const int kernel = idx.size(2);

    // some elements may not be updated, so zeros is needed
    auto output = torch::zeros({batch, Nkey, channels}, attn.options());

    f_attn_query2key<float, double>(
            at::cuda::getCurrentCUDAStream(),
            attn.data_ptr<float>(),
            query.data_ptr<float>(),
            idx.data_ptr<int>(),
            batch, Nquery, Nkey, kernel, channels,
            output.data_ptr<float>()
    );
    return output;
}
