<script>
  import {
    Progress,
    Table,
    Modal,
    ModalBody,
    ModalFooter,
    ModalHeader,
    Button,
  } from "sveltestrap";
  import { onMount } from "svelte";
  import t_jpg from "./assets/t.jpg";
  import beta_t_jpg from "./assets/beta_t.jpg";
  import alpha_t_jpg from "./assets/alpha_t.jpg";
  import alphas_cumprod_jpg from "./assets/alphas_cumprod.jpg";
  import alphas_cumprod_prev_jpg from "./assets/alphas_cumprod_prev.jpg";
  import sigma_jpg from "./assets/sigma.jpg";

  let loadingPercentage = 0;
  let canvas;
  let percentage = 0;
  let images = [];
  let parameters = null;
  let isModelOpen = false;
  const worker = new Worker("./64x64_cosin_300/worker.js?2");

  onMount(() => {
    worker.onmessage = async (evt) => {
      const data = evt.data;
      switch (data.type) {
        case "image": {
          if (!isModelOpen || data.timestep == 0) {
            updateCanvas(data.image);
          }

          percentage = data.percent;
          if (data.timestep == 0) {
            images.push(canvas.toDataURL());
            images = images;
          }
          break;
        }
        case "ready": {
          loadingPercentage = 100;
          parameters = data.parameters;
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

  const scale = 2;
  const updateCanvas = (img) => {
    const height = img[0].length;
    const width = img[0][0].length;

    canvas.width = width * scale;
    canvas.height = height * scale;
    const ctx = canvas.getContext("2d");
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
  };
</script>

<svelte:head>
  <link rel="stylesheet" href="./bootstrap/dist/css/bootstrap.min.css" />
</svelte:head>

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
            <canvas class="d-flex" bind:this={canvas} />
            <div
              class="d-flex progress_bar"
              style="width:{percentage * 100}%"
            />
            {#if parameters}
              <Button
                size="sm"
                color="secondary"
                on:click={() => (isModelOpen = true)}>Show Variables</Button
              >
            {/if}
          </div>
        </div>
      </div>
    </div>
  </div>
{/if}

<Modal
  isOpen={isModelOpen}
  toggle={() => (isModelOpen = false)}
  fullscreen={true}
>
  <ModalHeader toggle={() => (isModelOpen = false)}>Variables</ModalHeader>
  <ModalBody>
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
        {#each parameters.betas as beta, index}
          <tr>
            <th scope="row">{index + 1}</th>
            <td><code>{beta}</code></td>
            <td><code>{parameters.alphas[index].toFixed(19)}</code></td>
            <td><code>{parameters.alphas_cumprod[index].toFixed(19)}</code></td>
            <td
              ><code>{parameters.alphas_cumprod_prev[index].toFixed(19)}</code
              ></td
            >
            <td><code>{parameters.stddevs[index].toFixed(19)}</code></td>
          </tr>
        {/each}
      </tbody>
    </Table>
  </ModalBody>
  <ModalFooter>
    <Button color="secondary" on:click={() => (isModelOpen = false)}
      >Close</Button
    >
  </ModalFooter>
</Modal>

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

  :global(canvas) {
    width: 128px;
    height: 128px;
    border: solid 1px #666;
  }
</style>
