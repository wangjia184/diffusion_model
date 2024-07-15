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
    import ddim_png from "./assets/ddim.png";

    let loadingPercentage = 0;
    let canvas;
    let percentage = 0;
    let images = [];
    let variables = null;
    let skipSteps = 0;

    const worker = new Worker(
        "./64x64_cosin_300/worker.js?_" + new Date().getTime(),
    );

    $: skipSteps,
        (() => {
            worker.postMessage({ skipSteps: skipSteps });
        })();

    onMount(() => {
        const offscreen = canvas.transferControlToOffscreen();
        worker.postMessage({ offscreen: offscreen }, [offscreen]);
        worker.onmessage = async (evt) => {
            const data = evt.data;
            switch (data.type) {
                case "image": {
                    percentage = data.percent;
                    if (data.imageBlob) {
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            images.push(reader.result);
                            images = images;
                        };
                        reader.readAsDataURL(data.imageBlob);
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
</script>

<div class="border border-top-0 p-5">
    {#if loadingPercentage < 100}
        <div>
            <div class="text-center">Loading ...</div>
            <Progress animated color="success" value={loadingPercentage}
                >{loadingPercentage}%</Progress
            >
        </div>
    {:else}
        <div class="mb-3 row">
            <label for="acc" class="col-sm-2 text-end">Acceleration :</label>
            <div class="col-sm-8">
                <input
                    type="range"
                    class="form-range"
                    min="0"
                    max="20"
                    step="1"
                    id="acc"
                    bind:value={skipSteps}
                />
            </div>
            <div class="col-sm-2 text-start">
                <small>
                    <strong
                        >{skipSteps <= 0
                            ? "DDPM"
                            : "DDIM; m=n-" + skipSteps.toString()}</strong
                    >
                </small>
            </div>
        </div>
    {/if}
    <h1>
        <img src={ddim_png} alt="" style="display: {skipSteps ? '' : 'none'}" />
        <img
            src={reverse_jpg}
            alt=""
            style="display: {skipSteps ? 'none' : ''}"
        />
    </h1>
    <hr />
    <div class="container mt-2">
        <div class="row">
            {#each images as img}
                <div class="col">
                    <img src={img} class="generated mb-2" alt="" />
                </div>
            {/each}
            <div class="col">
                <div class="d-flex justify-content-center">
                    <div>
                        <canvas
                            class="d-flex avatar"
                            bind:this={canvas}
                            width="128"
                            height="128"
                        />
                        <div
                            class="d-flex progress_bar"
                            style="width:{percentage * 100}%"
                        />
                    </div>
                </div>
            </div>
        </div>
    </div>
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
                                    19,
                                )}</code
                            ></td
                        >
                        <td
                            ><code
                                >{variables.alphas_cumprod_prev[index].toFixed(
                                    19,
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
