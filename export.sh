tensorflowjs_converter --version
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --saved_model_tags=serve \
    --quantize_float16=* \
    ./saved_model \
    ./web/public/64x64_cosin_300


#--weight_shard_size_bytes=1048576 \