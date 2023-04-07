<script>
    import { onMount } from "svelte";
    import singlestep_jpg from "./assets/singlestep.jpg";
    import monnalisa_jpg from "./assets/monnalisa.jpg";

    const worker = new Worker("./singlestep.js");
    let canvas;
    let image_data;
    onMount(() => {
        const img = new Image();
        img.src = monnalisa_jpg;
        img.onload = () => {
            const ctx = new OffscreenCanvas(img.width, img.height).getContext(
                "2d"
            );
            ctx.drawImage(img, 0, 0);
            image_data = ctx.getImageData(0, 0, img.width, img.height);
            worker.postMessage({
                image_data: image_data,
                x: 0,
                y: 0,
            });
            worker.onmessage = (evt) => {
                canvas.width = evt.data.width;
                canvas.height = evt.data.height;
                canvas.getContext("2d").putImageData(evt.data.image_data, 0, 0);
            };
        };
    });

    const handleMousemove = (evt) => {
        worker.postMessage({
            image_data: image_data,
            x: evt.offsetX,
            y: evt.offsetY,
        });
    };
</script>

<h1><img src={singlestep_jpg} alt="x=" /></h1>

<canvas bind:this={canvas} class="canvas" on:mousemove={handleMousemove} />

<style>
    .canvas {
        border: 1px solid #ccc;
    }
</style>
