<script>
    import { Icon, Table } from "sveltestrap";
    import { onMount } from "svelte";
    import xt_jpg from "./assets/xt.jpg";
    import monnalisa_jpg from "./assets/monnalisa.jpg";

    import t_jpg from "./assets/t.jpg";
    import beta_t_jpg from "./assets/beta_t.jpg";
    import alpha_t_jpg from "./assets/alpha_t.jpg";
    import alphas_cumprod_jpg from "./assets/alphas_cumprod.jpg";
    import sqrt_alphabar_jpg from "./assets/sqrt_alphabar.jpg";
    import sqrt_one_minus_alphabar_jpg from "./assets/sqrt_one_minus_alphabar.jpg";

    const worker = new Worker("./multistep.js");
    let canvas;
    let images = [
        {
            timestep: 0,
            src: monnalisa_jpg,
        },
    ];
    let variables = {};
    onMount(() => {
        const img = new Image();
        img.src = monnalisa_jpg;
        img.onload = () => {
            const ctx = new OffscreenCanvas(img.width, img.height).getContext(
                "2d"
            );
            const IMG_SIZE = 150;
            ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);
            worker.postMessage({
                image_data: ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE),
            });
            worker.onmessage = (evt) => {
                const image_data = evt.data.image_data;
                if (image_data) {
                    canvas.width = evt.data.image_data.width;
                    canvas.height = evt.data.image_data.height;
                    canvas
                        .getContext("2d")
                        .putImageData(evt.data.image_data, 0, 0);
                    images.push({
                        timestep: evt.data.timestep,
                        src: canvas.toDataURL(),
                    });
                    images = images;
                }
                if (evt.data.variables) {
                    variables = evt.data.variables;
                }
            };
        };
    });
</script>

<h1><img src={xt_jpg} alt="" /></h1>
<hr />
<div class="d-flex flex-wrap justify-content-center">
    {#each images as image}
        <div class="m-1">
            <img src={image.src} class="image" alt="" />
            <h2>t = {image.timestep}</h2>
        </div>
    {/each}
</div>

<canvas bind:this={canvas} class="canvas" style="display:hidden" />
<br />
<a href="multistep.js" target="_blank" class="link-info"
    ><Icon name="filetype-js" /> Check Source Code</a
>

<hr />
<Table bordered>
    <thead style="position:sticky; top: 0;">
        <tr>
            <th style="background-color:white"
                ><img alt="timestep" src={t_jpg} /></th
            >
            <th style="background-color:white"
                ><img alt="beta t" src={beta_t_jpg} /></th
            >
            <th style="background-color:white"
                ><img alt="alpha t" src={alpha_t_jpg} /></th
            >
            <th style="background-color:white"
                ><img alt="alpha bar t" src={alphas_cumprod_jpg} /></th
            >
            <th style="background-color:white"
                ><img alt="alpha bar t minus one" src={sqrt_alphabar_jpg} /></th
            >
            <th style="background-color:white"
                ><img alt="std dev" src={sqrt_one_minus_alphabar_jpg} /></th
            >
        </tr>
    </thead>
    <tbody>
        {#if variables && variables.betas}
            {#each variables.betas as beta, index}
                <tr>
                    <th scope="row">{index + 1}</th>
                    <td><code>{beta}</code></td>
                    <td><code>{variables.alphas[index].toFixed(19)}</code></td>
                    <td
                        ><code
                            >{variables.alphas_cumprod[index].toFixed(19)}</code
                        ></td
                    >
                    <td
                        ><code
                            >{variables.sqrt_alphas_cumprod[index].toFixed(
                                19
                            )}</code
                        ></td
                    >
                    <td
                        ><code
                            >{variables.sqrt_one_minus_alphas_cumprod[
                                index
                            ].toFixed(19)}</code
                        ></td
                    >
                </tr>
            {/each}
        {/if}
    </tbody>
</Table>

<style>
    .image {
        width: 150px;
        height: 150px;
    }
    .canvas {
        display: none;
    }
</style>
