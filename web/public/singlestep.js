let x_img = null;
let noise = null;

const CANVAS_WIDTH = 1200;
const CANVAS_HEIGHT = 800;
const DIAMETER = 300;
const center = CANVAS_WIDTH / 2;
const left = (CANVAS_WIDTH - DIAMETER) / 2;
const right = left + DIAMETER;
const bottom = CANVAS_HEIGHT - 400;

let theta = Math.PI / 3;

self.onmessage = (evt) => {

    if (!noise || !x_img) {
        if (!evt.data.image_data)
            return;
        x_img = new OffscreenCanvas(evt.data.image_data.width, evt.data.image_data.height);
        x_img.getContext("2d").putImageData(evt.data.image_data, 0, 0);
        noise = generateNoiseImage(x_img.width, x_img.height);
        refresh();
    }

    const y = Math.min(evt.data.y, bottom);
    theta = Math.atan2(bottom - y, center - evt.data.x);



};

const refresh = () => {
    const offscreen = draw(theta);
    self.postMessage({
        image_data: offscreen.getContext("2d").getImageData(0, 0, offscreen.width, offscreen.height, {
            colorSpace: "srgb",
        }),
        width: offscreen.width,
        height: offscreen.height
    });
    setTimeout(refresh, 0);
};




const draw = (theta) => {






    const offscreen = new OffscreenCanvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    const ctx = offscreen.getContext("2d");
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

    const x = center - (Math.cos(theta) * DIAMETER) / 2;
    const y = bottom - (Math.sin(theta) * DIAMETER) / 2;

    ctx.lineWidth = 1;
    ctx.strokeStyle = "#666";

    ctx.beginPath();
    ctx.arc(center, bottom, DIAMETER / 2, Math.PI, 0);
    ctx.stroke();

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#666";
    ctx.moveTo(left, bottom);
    ctx.lineTo(right, bottom);

    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(left, bottom);
    ctx.lineTo(x, y);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#00CCCC";
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(right, bottom);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#CC00CC";
    ctx.stroke();

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.PI - (Math.PI - theta) / 2);
    ctx.globalAlpha = 0.2;
    ctx.fillStyle = "rgb( 0, 200, 200, 100)";
    const sqrt_beta = Math.sin(0.5 * theta);
    ctx.fillRect(0, 0, sqrt_beta * DIAMETER, sqrt_beta * DIAMETER);
    ctx.restore();

    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(2 * Math.PI - (Math.PI - theta) / 2);
    ctx.globalAlpha = 0.2;
    ctx.fillStyle = "rgb( 200, 0, 200, 100)";
    const sqrt_one_minus_beta = Math.cos(0.5 * theta);
    ctx.fillRect(
        0,
        0,
        sqrt_one_minus_beta * DIAMETER,
        sqrt_one_minus_beta * DIAMETER
    );
    ctx.restore();

    ctx.font = "24px Verdana";
    ctx.textAlign = "center";
    const text =
        "β=" +
        Math.pow(sqrt_beta, 2).toFixed(8) +
        "; 1-β=" +
        Math.pow(sqrt_one_minus_beta, 2).toFixed(8) +
        ";";
    ctx.fillText(text, center, 30);

    const img_y = CANVAS_HEIGHT - x_img.height - 20;
    const text_y = img_y - 20;
    // left image
    ctx.drawImage(noise, 20, img_y);
    const text_sqrt_beta = "sqrt(β)=" + sqrt_beta.toFixed(8);
    ctx.textAlign = "center";
    ctx.fillText(
        text_sqrt_beta,
        20 + x_img.width / 2,
        text_y
    );

    // center image
    ctx.drawImage(mix(x_img, noise, sqrt_beta, sqrt_one_minus_beta), center - noise.width / 2, img_y);



    // right image
    const text_sqrt_one_minus_beta =
        "sqrt(1-β)=" + sqrt_one_minus_beta.toFixed(8);
    ctx.textAlign = "center";
    ctx.fillText(text_sqrt_one_minus_beta, CANVAS_WIDTH - x_img.width / 2 - 20, text_y);
    ctx.drawImage(
        x_img,
        CANVAS_WIDTH - x_img.width - 20,
        img_y
    );

    return offscreen;

};

const generateNoiseImage = (width, height) => {
    // Standard Normal variate using Box-Muller transform.
    const generateGaussianNoise = (mean = 0, stdev = 1) => {
        let u = 1 - Math.random(); // Converting [0,1) to (0,1]
        let v = Math.random();
        let z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
        // Transform to the desired mean and standard deviation:
        return z * stdev + mean;
    };

    const offscreen = new OffscreenCanvas(width, height);
    const ctx = offscreen.getContext("2d");

    const id = ctx.createImageData(width, height, {
        colorSpace: "srgb",
    });
    const pixels = id.data;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const r = Math.max(
                0,
                Math.min(255, generateGaussianNoise() * 127.5 + 127.5)
            );
            const g = Math.max(
                0,
                Math.min(255, generateGaussianNoise() * 127.5 + 127.5)
            );
            const b = Math.max(
                0,
                Math.min(255, generateGaussianNoise() * 127.5 + 127.5)
            );

            const offset = (y * width + x) * 4;

            pixels[offset] = r;
            pixels[offset + 1] = g;
            pixels[offset + 2] = b;
            pixels[offset + 3] = 255;
        }
    }

    ctx.putImageData(id, 0, 0);

    return offscreen;
};

const mix = (x, epsilon, sqrt_beta, sqrt_one_minus_beta) => {
    const offscreen = new OffscreenCanvas(
        x_img.width,
        x_img.height
    );
    const ctx = offscreen.getContext("2d");

    const id = ctx.createImageData(x_img.width, x_img.height, {
        colorSpace: "srgb",
    });
    const pixels = id.data;

    const src1 = x.getContext("2d").getImageData(0, 0, x.width, x.height, {
        colorSpace: "srgb",
    }).data;

    const src2 = epsilon
        .getContext("2d")
        .getImageData(0, 0, epsilon.width, epsilon.height, {
            colorSpace: "srgb",
        }).data;

    for (let y = 0; y < x_img.height; y++) {
        for (let x = 0; x < x_img.width; x++) {
            const offset = (y * x_img.width + x) * 4;
            for (let index = 0; index < 3; index++) {
                const ch = Math.max(
                    0,
                    Math.min(
                        255,
                        sqrt_one_minus_beta * src1[offset + index] +
                        sqrt_beta * src2[offset + index]
                    )
                );

                pixels[offset + index] = ch;
            }
            pixels[offset + 3] = 255;
        }
    }

    ctx.putImageData(id, 0, 0);

    return offscreen;
};