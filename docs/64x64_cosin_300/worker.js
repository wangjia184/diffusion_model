

const url = self.location.toString();
let index = url.lastIndexOf('/');
index = url.lastIndexOf('/', index - 1);

const TF_JS_URL = url.substring(0, index) + "/@tensorflow/tfjs/dist/tf.min.js";
const TF_JS_CDN_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.2.0/dist/tf.min.js";
async function load_model() {

    const model = await (async () => {
        try {
            self.postMessage({ type: 'progress', progress: 0.1, message: 'Loading model' });
            return await tf.loadGraphModel('./model.json', {
                onProgress: (percent) => {
                    self.postMessage({ type: 'progress', progress: 0.1 + 0.9 * percent, message: 'Loading model' });
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


    return model;
}

async function main() {

    try {
        await tf.setBackend('webgl');
        console.log('Successfully loaded WebGL backend');
    } catch {
        await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.2.0/dist/tf-backend-wasm.min.js');
        await tf.setBackend('wasm');
        console.log('Successfully loaded WASM backend');
    }



    const model = await load_model();

    const stable_sqrt = (number) => number > 0 && number < 0.001 ? Math.exp(0.5 * Math.log(number)) : Math.sqrt(number);

    // cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    const cosine_beta_schedule = (timesteps, s) => {
        const steps = timesteps + 1.0;
        let alphas_cumprod = tf.linspace(0, timesteps, steps).arraySync().map(x => {
            return Math.pow(Math.cos(((x / timesteps) + s) / (1 + s) * Math.PI * 0.5), 2);
        });
        const base = alphas_cumprod[0];
        alphas_cumprod = alphas_cumprod.map(x => x / base);
        const betas = new Array(timesteps);
        for (let index = 0; index < betas.length; index++) {
            const b = 1 - (alphas_cumprod[index + 1] / alphas_cumprod[index]);
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
    const stddevs = new Array(timesteps);

    alphas_cumprod_prev[0] = 1.0;

    // prepare variables
    betas.forEach((beta, index) => {
        alphas[index] = 1.0 - beta;

        alphas_cumprod[index] = (index > 0) ? alphas_cumprod[index - 1] * alphas[index] : alphas[index];

        if (index < timesteps - 1) {
            alphas_cumprod_prev[index + 1] = alphas_cumprod[index];
        }

        stddevs[index] = stable_sqrt(beta * (1.0 - alphas_cumprod_prev[index]) / (1.0 - alphas_cumprod[index]))
    });


    self.postMessage({
        type: 'ready',
        parameters: {
            betas: betas,
            alphas: alphas,
            alphas_cumprod: alphas_cumprod,
            alphas_cumprod_prev: alphas_cumprod_prev,
            stddevs: stddevs
        }
    });

    const ddpm_p_sample = (xt, timeStep) => {
        // When using WebGL backend, tf.Tensor memory must be managed explicitly (it is not sufficient to let a tf.Tensor go out of scope for its memory to be released).
        // Here we use an array to collect all tensors to be disposed when this method exits
        const collection = new Array();

        const time_input = tf.tensor(timeStep, [1]/*shape*/, 'int32' /* model.signature.inputs.time_input.dtype */);
        collection.push(time_input);

        const inputs = {
            time_input: time_input,
            image_input: xt
        };

        const epsilon = model.predict(inputs);
        collection.push(epsilon);

        const epsilon2 = epsilon.mul(stable_sqrt(1 - alphas_cumprod[timeStep]));
        collection.push(epsilon2);

        const xt_sub_epsilon2 = xt.sub(epsilon2);
        collection.push(xt_sub_epsilon2);

        const x0 = xt_sub_epsilon2.div(stable_sqrt(alphas_cumprod[timeStep]));
        collection.push(x0);

        const clipped_x0 = tf.clipByValue(x0, -1.0, 1.0);
        collection.push(clipped_x0);

        const x0_coefficient = stable_sqrt(alphas_cumprod_prev[timeStep]) * betas[timeStep] / (1 - alphas_cumprod[timeStep]);
        const x0_multipled_by_coef = clipped_x0.mul(x0_coefficient);
        collection.push(x0_multipled_by_coef);

        const xt_coefficient = stable_sqrt(alphas[timeStep]) * (1 - alphas_cumprod_prev[timeStep]) / (1 - alphas_cumprod[timeStep]);
        const xt_multipled_by_coef = xt.mul(xt_coefficient);
        collection.push(xt_multipled_by_coef);

        const mean = x0_multipled_by_coef.add(xt_multipled_by_coef);
        collection.push(mean);

        let normal_noise = tf.randomNormal(xt.shape, 0/*mean*/, 1/*stddev*/, 'float32', Math.random() * 10000/*seed*/);
        collection.push(normal_noise);

        normal_noise = normal_noise.mul(stddevs[timeStep]);
        collection.push(normal_noise);

        const xt_minus_one = mean.add(normal_noise);

        collection.forEach((t) => t.dispose());
        return xt_minus_one;
    };// ddpm_p_sample()

    const scale = 2;
    const offscreen = new OffscreenCanvas(image_size * scale, image_size * scale);
    const get_image = (img) => {
        const height = img[0].length;
        const width = img[0][0].length;

        const ctx = offscreen.getContext("2d");
        const id = ctx.createImageData(width * scale, height * scale, {
            colorSpace: "srgb",
        });
        const pixels = id.data;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const channels = img[0][y][x];

                const r = Math.max(0, Math.min(255, channels[0] * 127.5 + 127.5));
                const g = Math.max(0, Math.min(255, channels[1] * 127.5 + 127.5));
                const b = Math.max(0, Math.min(255, channels[2] * 127.5 + 127.5));

                for (let offsetX = 0; offsetX < scale; offsetX++) {
                    for (let offsetY = 0; offsetY < scale; offsetY++) {
                        const offset =
                            ((y * scale + offsetY) * width * scale + x * scale + offsetX) * 4;

                        pixels[offset] = r;
                        pixels[offset + 1] = g;
                        pixels[offset + 2] = b;
                        pixels[offset + 3] = 255;
                    }
                }
            }
        }

        ctx.putImageData(id, 0, 0);

        return offscreen.transferToImageBitmap();
    };// produce_image()

    for (; ;) {
        let timestep = timesteps - 1;
        const shape = [1, image_size, image_size, 3];
        let xt = tf.randomNormal(shape, 0/*mean*/, 1/*stddev*/, 'float32', Math.random() * 10000/*seed*/);
        while (timestep >= 0) {
            const xt_minus_one = ddpm_p_sample(xt, timestep);
            xt.dispose();
            xt = xt_minus_one;

            self.postMessage({
                type: 'image',
                timestep: timestep,
                image: get_image(xt_minus_one.arraySync()),
                percent: (timesteps - timestep) / (1.0 * timesteps)
            });
            timestep--;
        }
        xt.dispose();
    }








}


self.postMessage({ type: 'progress', progress: 0, message: 'Loading ' + TF_JS_URL });
import(TF_JS_URL)
    .then(async () => {

        await main();

    })
    .catch(async (err) => {

        try {
            await import(TF_JS_CDN_URL);
            await main();
        }
        catch {
            self.postMessage({ type: 'error', message: 'Unable to load ' + TF_JS_URL });
            console.log('Unable to load ' + TF_JS_URL, err);
        }
    });


