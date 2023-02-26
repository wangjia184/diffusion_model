

const url = self.location.toString();
let index = url.lastIndexOf('/');
index = url.lastIndexOf('/', index-1);
console.log(url, index);
const TF_JS_URL = url.substring( 0, index) + "/@tensorflow/tfjs@4.2.0/dist/tf.min.js";

console.log(TF_JS_URL);

async function load_model() {
    try {
        //await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.2.0/dist/tf-backend-webgl.min.js');
        //await tf.setBackend('webgl');

        //tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.2.0/dist/')
        //const backend = await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.2.0/dist/tf-backend-wasm.min.js');
        //tf.wasm.setWasmPaths('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.2.0/dist/');
        

        

        
        self.postMessage({ type: 'progress', progress: 0.3, message: 'Loading model' });
    } catch (e) {
        self.postMessage({ type: 'error', message: 'Failed to load tensorflow WASM backend' });
        console.log('Failed to initialize tensorflow backend', e);
        return null;
    }

    const model = await (async () => {
        try {
            self.postMessage({ type: 'progress', progress: 0.1, message: 'Loading model' });
            return await tf.loadGraphModel('./model.json', {
                onProgress : (percent) => {
                    self.postMessage({ type: 'progress', progress: 0.1 + 0.9*percent, message: 'Loading model' });
                }
            });
        }
        catch (e) {
            self.postMessage({ type: 'error', message: 'Failed to load model.' + e });
            console.log('Failed to load model', e);
            return null;
        }
    })();

    if (!model) {
        self.postMessage({ type: 'error', message: 'Failed to load model' });
        return;
    }

    self.postMessage({ type: 'ready' });
    return model;
}



self.postMessage({ type: 'progress', progress: 0, message: 'Loading ' + TF_JS_URL });
import(TF_JS_URL)
    .then(async () => {

        const model = await load_model();

        const stable_sqrt = (number) => Math.sqrt(number); // Math.exp( 0.5 * Math.log(Math.max(number, 1e-20)) );

        // cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        const cosine_beta_schedule = (timesteps, s) => {
            const steps = timesteps + 1.0;
            let alphas_cumprod = tf.linspace(0, timesteps, steps).arraySync().map( x => {
                return Math.pow( Math.cos(((x / timesteps) + s) / (1 + s) * Math.PI * 0.5), 2);
            });
            const base = alphas_cumprod[0];
            alphas_cumprod = alphas_cumprod.map( x => x / base);
            const betas = new Array(timesteps);
            for( let index = 0; index < betas.length; index++){
                const b = 1 - (alphas_cumprod[index+1] / alphas_cumprod[index]);
                betas[index] = Math.min(0.9999, Math.max(0.0001, b));
            }
            return betas;
        };
    
        
        const image_size = 64;
        const timesteps = 300;
        //const betas = tf.linspace( beta_start, beta_end, timesteps).arraySync();
        const betas = cosine_beta_schedule(timesteps, 0.008);
        const alphas = new Array(timesteps);
        const alphas_cumprod = new Array(timesteps);
        const alphas_cumprod_prev = new Array(timesteps);
        const sqrt_one_minus_alphas_cumprod = new Array(timesteps);
        const sqrt_recip_alphas_cumprod = new Array(timesteps);
        const sqrt_recipm1_alphas_cumprod = new Array(timesteps);
        const stddevs = new Array(timesteps);

        alphas_cumprod_prev[0] = 1.0;

        // prepare variables
        betas.forEach( (beta, index) => {
            alphas[index] = 1.0 - beta;

            alphas_cumprod[index] = (index > 0) ? alphas_cumprod[index-1] * alphas[index] : alphas[index];

            if( index < timesteps - 1 ){
                alphas_cumprod_prev[index+1] = alphas_cumprod[index];
            }

            sqrt_recip_alphas_cumprod[index] = stable_sqrt(1.0 / alphas_cumprod[index]);
            sqrt_recipm1_alphas_cumprod[index] = stable_sqrt(1.0 / alphas_cumprod[index] - 1);

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
        

        const ddpm_p_sample = (image_input, timeStep) => {
            // When using WebGL backend, tf.Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
            // Here we use an array to collect all tensors to be disposed when this method exits
            const collection = new Array();
            
            const time_input = tf.tensor(timeStep, [1]/*shape*/, 'int32' /* model.signature.inputs.time_input.dtype */);
            collection.push(time_input);

            const inputs = {
                time_input : time_input,
                image_input : image_input
            };

            const epsilon = model.predict(inputs);
            collection.push(epsilon);

            const epsilon2 = epsilon.mul(Math.sqrt(1 - alphas_cumprod[timeStep]));
            collection.push(epsilon2);

            const xt_sub_epsilon2 = image_input.sub(epsilon2);
            collection.push(xt_sub_epsilon2);

            const x0 = xt_sub_epsilon2.div(Math.sqrt(alphas_cumprod[timeStep]));
            collection.push(x0);

            const clipped_x0 = tf.clipByValue(x0, -1.0, 1.0);
            collection.push(clipped_x0);

            const x0_coefficient = Math.sqrt(alphas_cumprod_prev[timeStep]) * betas[timeStep] / (1 - alphas_cumprod[timeStep]);
            const x0_multipled_by_coef = clipped_x0.mul(x0_coefficient); 
            collection.push(x0_multipled_by_coef);
    
            const xt_coefficient = Math.sqrt(alphas[timeStep]) * (1 - alphas_cumprod_prev[timeStep]) / (1 - alphas_cumprod[timeStep]);
            const xt_multipled_by_coef = image_input.mul(xt_coefficient);
            collection.push(xt_multipled_by_coef);

            const mean = x0_multipled_by_coef.add(xt_multipled_by_coef);
            collection.push(mean);

            let normal_noise = tf.randomNormal(image_input.shape, 0/*mean*/, 1/*stddev*/, 'float32', Math.random()*10000/*seed*/);
            collection.push(normal_noise);

            normal_noise = normal_noise.mul(stddevs[timeStep]); 
            collection.push(normal_noise);

            const xt_minus_one = mean.add(normal_noise); 
  
            collection.forEach( (t) => t.dispose() );
            return xt_minus_one;
        };

        const map = {};
        let keySource = 0;
        self.onmessage = (evt) => {
            const request = evt.data;

            switch (request.type) {
                case 'ddpmStart': {
                    const timestep = timesteps - 1;
                    const shape = [1, image_size, image_size, 3];
                    const initial_noises = tf.randomNormal(shape, 0/*mean*/, 1/*stddev*/, 'float32', Math.random()*10000/*seed*/);
                    const image = ddpm_p_sample(initial_noises, timestep);
                    initial_noises.dispose();
                    const key = ++keySource;
                    map[key] = {
                        step : timestep,
                        image : image
                    };
                    
                    self.postMessage({ type: 'reply', id: request.id, data: {
                        step : timestep,
                        image : image.arraySync(),
                        key : key,
                        percent : (timesteps - timestep) / (1.0 * timesteps),
                    } });
                    break;
                }

                case 'ddpmNext': {
                    const prev = map[request.kwargs && request.kwargs.key];
                    if(!prev) {
                        console.log('invalid request');
                    } else {
                        delete map[request.kwargs.key];
                        const timestep = prev.step - 1;
                        const image = ddpm_p_sample(prev.image, timestep);
                        prev.image.dispose();
                        if(timestep > 0) {
                            map[request.kwargs.key] = {
                                step : timestep,
                                image : image
                            };
                        }
                        self.postMessage({ type: 'reply', id: request.id, data: {
                            step : timestep,
                            image : image.arraySync(),
                            key : request.kwargs.key,
                            percent : (timesteps - timestep) / (1.0 * timesteps)
                        } });
                    }
                    break;
                }
        
            }
        };

        

    



    })
    .catch((err) => {
        self.postMessage({ type: 'error', message: 'Unable to load tensorflow.js' });
        console.log('Unable to load tensorflow.js', err);
    });


