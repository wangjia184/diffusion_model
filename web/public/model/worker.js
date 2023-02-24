

self.postMessage({ type: 'progress', progress: 0, message: 'Loading tensorflow' });

async function load_model() {
    // successfully loaded tensorflow.js
    self.postMessage({ type: 'progress', progress: 0.1, message: 'Loading tensorflow WASM backend' });

    try {
        const backend = await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.2.0/dist/tf-backend-webgl.min.js');

        //const backend = await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.2.0/dist/tf-backend-wasm.min.js');
        //tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.2.0/dist/');
        //await tf.setBackend('wasm');

        
        await tf.setBackend('webgl');
        self.postMessage({ type: 'progress', progress: 0.3, message: 'Loading model' });
    } catch (e) {
        self.postMessage({ type: 'error', message: 'Failed to load tensorflow WASM backend' });
        console.log('Failed to initialize tensorflow backend', e);
        return null;
    }

    const model = await (async () => {
        try {
            return await tf.loadGraphModel('./model.json');
        }
        catch (e) {
            console.log('Failed to load model', e);
            return null;
        }
    })();

    if (!model) {
        self.postMessage({ type: 'error', message: 'Failed to load model' });
        return;
    } else {
        self.postMessage({ type: 'progress', progress: 0.6, message: 'Loading WASM application' });
    }
    
    return model;
}

import("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.2.0/dist/tf.min.js")
    .then(async () => {

        const model = await load_model();

        const image_size = 64;
        const beta_start = 1e-4;
        const beta_end = 0.02;
        const timesteps = 1000;
        const betas = tf.linspace( beta_start, beta_end, timesteps).arraySync();
        const alphas = new Array(timesteps);
        const alphas_cumprod = new Array(timesteps);
        const alphas_cumprod_prev = new Array(timesteps);
        const sqrt_one_minus_alphas_cumprod = new Array(timesteps);
        const stddevs = new Array(timesteps);

        alphas_cumprod_prev[0] = 1.0;

        // prepare variables
        betas.forEach( (beta, index) => {
            alphas[index] = 1.0 - beta;

            alphas_cumprod[index] = (index > 0) ? alphas_cumprod[index-1] * alphas[index] : alphas[index];

            if( index < timesteps - 1 ){
                alphas_cumprod_prev[index+1] = alphas_cumprod[index];
            }

            sqrt_one_minus_alphas_cumprod[index] = Math.sqrt(1.0 - alphas_cumprod[index]);

            const variance = beta * (1.0 - alphas_cumprod_prev[index]) / (1.0 - alphas_cumprod[index]);
            stddevs[index] = Math.exp( 0.5 * Math.log(Math.max(variance, 1e-20)) ); // Log calculation clipped because the posterior variance is 0 at the beginning 
        });

        if( typeof(console.log) === 'function' ){
            console.log("betas=", betas);
            console.log("alphas=", alphas);
            console.log("alphas_cumprod=", alphas_cumprod);
            console.log("alphas_cumprod_prev=", alphas_cumprod_prev);
            console.log("sqrt_one_minus_alphas_cumprod=", sqrt_one_minus_alphas_cumprod);
            console.log("stddevs=", stddevs);
        }
        

        const p_sample = (image_input, timeStep) => {
            // When using WebGL backend, tf.Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
            // Here we use an array to collect all tensors to be disposed when this method exits
            const collection = new Array();
            
            const time_input = tf.tensor(timeStep, [1]/*shape*/, 'int32' /* model.signature.inputs.time_input.dtype */);
            collection.push(time_input);

            const inputs = {
                time_input : time_input,
                image_input : image_input
            };

            let tensor = model.predict(inputs);
            collection.push(tensor);

            const coefficient = betas[timeStep] / sqrt_one_minus_alphas_cumprod[timeStep];
            tensor = tensor.mul(coefficient); 
            collection.push(tensor);
    
            tensor = image_input.sub(tensor); 
            collection.push(tensor);

            tensor = tensor.div(Math.sqrt(alphas[timeStep])); 
            collection.push(tensor);

            let normal_noise = tf.randomNormal(tensor.shape, 0/*mean*/, 1/*stddev*/, 'float32', Math.random()*10000/*seed*/);
            collection.push(normal_noise);

            normal_noise = normal_noise.mul(stddevs[timeStep]); 
            collection.push(normal_noise);

            tensor = tensor.add(normal_noise); 
  
            collection.forEach( (t) => t.dispose() );
            return tensor;
        };

        
        const shape = [1, image_size, image_size, 3];
        const initial_noises = tf.randomNormal(shape, 0/*mean*/, 1/*stddev*/, 'float32', Math.random()*10000/*seed*/);

        let image = initial_noises;
        for( let timestep = timesteps - 1; timestep >= 0; timestep--){
            const new_image = p_sample(image, timestep);
            image.dispose();
            image = new_image;

            self.postMessage({ type: 'data', image: image.arraySync(), timestep : timestep });
            
        }
        image.dispose();
    



    })
    .catch((err) => {
        self.postMessage({ type: 'error', message: 'Unable to load tensorflow.js' });
        console.log('Unable to load tensorflow.js', err);
    });


