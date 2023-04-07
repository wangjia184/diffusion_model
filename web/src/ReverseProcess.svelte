<script>
    import { Table, Progress, Icon } from "sveltestrap";
    import { onMount } from "svelte";
    import t_jpg from "./assets/t.jpg";
    import beta_t_jpg from "./assets/beta_t.jpg";
    import alpha_t_jpg from "./assets/alpha_t.jpg";
    import alphas_cumprod_jpg from "./assets/alphas_cumprod.jpg";
    import alphas_cumprod_prev_jpg from "./assets/alphas_cumprod_prev.jpg";
    import sigma_jpg from "./assets/sigma.jpg";
    import reverse_jpg from "./assets/reverse.jpg";

    let loadingPercentage = 0;
    let canvas;
    let percentage = 0;
    let images = [];
    let variables = null;

    const worker = new Worker("./64x64_cosin_300/worker.js?4");

    onMount(() => {
        worker.onmessage = async (evt) => {
            const data = evt.data;
            switch (data.type) {
                case "image": {
                    updateCanvas(data.image);

                    percentage = data.percent;
                    if (data.timestep == 0) {
                        images.push(canvas.toDataURL());
                        images = images;
                    }
                    break;
                }
                case "ready": {
                    loadingPercentage = 100;
                    variables = data.parameters;
                    break;
                }
                case "progress": {
                    loadingPercentage = Math.floor(data.progress * 100);
                    break;
                }
                case "error": {
                    alert(data.message);
                    self.location = self.location;
                    break;
                }
                default: {
                    console.log(data);
                    break;
                }
            }
        };
    });

    const updateCanvas = (img) => {
        const ctx = canvas.getContext("bitmaprenderer");
        ctx.transferFromImageBitmap(img);
    };
</script>

<div class="border border-top-0 p-5">
    <h1><img src={reverse_jpg} alt="" /></h1>
    <hr />
    {#if loadingPercentage < 100}
        <div>
            <div class="text-center">Loading ...</div>
            <Progress animated color="success" value={loadingPercentage}
                >{loadingPercentage}%</Progress
            >
        </div>
    {:else}
        <div class="container">
            <div class="row">
                {#each images as img}
                    <div class="col">
                        <img src={img} class="generated mb-2" alt="" />
                    </div>
                {/each}
                <div class="col">
                    <div class="d-flex justify-content-center">
                        <div>
                            <canvas class="d-flex avatar" bind:this={canvas} />
                            <div
                                class="d-flex progress_bar"
                                style="width:{percentage * 100}%"
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    {/if}
    <a href="./64x64_cosin_300/worker.js" target="_blank" class="link-info"
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
                    ><img
                        alt="alpha bar t minus one"
                        src={alphas_cumprod_prev_jpg}
                    /></th
                >
                <th style="background-color:white"
                    ><img alt="std dev" src={sigma_jpg} /></th
                >
            </tr>
        </thead>
        <tbody>
            {#if variables && variables.betas}
                {#each variables.betas as beta, index}
                    <tr>
                        <th scope="row">{index + 1}</th>
                        <td><code>{beta}</code></td>
                        <td
                            ><code>{variables.alphas[index].toFixed(19)}</code
                            ></td
                        >
                        <td
                            ><code
                                >{variables.alphas_cumprod[index].toFixed(
                                    19
                                )}</code
                            ></td
                        >
                        <td
                            ><code
                                >{variables.alphas_cumprod_prev[index].toFixed(
                                    19
                                )}</code
                            ></td
                        >
                        <td
                            ><code>{variables.stddevs[index].toFixed(19)}</code
                            ></td
                        >
                    </tr>
                {/each}
            {/if}
        </tbody>
    </Table>
</div>

<style>
    .progress_bar {
        height: 3px;
        background-color: darkorange;
    }

    .generated {
        width: 128px;
        height: 128px;
        border: solid 1px #666;
    }

    .avatar {
        width: 128px;
        height: 128px;
        border: solid 1px #666;
    }
</style>
