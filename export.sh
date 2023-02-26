tensorflowjs_converter --version

# https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/README.md
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --saved_model_tags=serve \
    --quantize_uint8=* \
    --weight_shard_size_bytes=1048576 \
    ./saved_model \
    ./web/public/64x64_cosin_300
