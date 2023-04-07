<script>
  import { Progress, Button } from "sveltestrap";
  import { onMount } from "svelte";
  import VariableDlg from "./Variables.svelte";
  import SingleStep from "./SingleStep.svelte";

  let loadingPercentage = 0;
  let canvas;
  let percentage = 0;
  let images = [];
  let parameters = null;
  let isVariableModelOpen = false;
  const worker = new Worker("./64x64_cosin_300/worker.js?3");

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

  const updateCanvas = (img) => {
    const ctx = canvas.getContext("bitmaprenderer");
    ctx.transferFromImageBitmap(img);
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
            <canvas class="d-flex avatar" bind:this={canvas} />
            <div
              class="d-flex progress_bar"
              style="width:{percentage * 100}%"
            />
            {#if parameters}
              <Button
                size="sm"
                color="secondary"
                on:click={() => (isVariableModelOpen = true)}
                >Show Variables</Button
              >
            {/if}
          </div>
        </div>
      </div>
    </div>
  </div>
  <br />
  <hr />
  <SingleStep />
{/if}

<VariableDlg bind:open={isVariableModelOpen} variables={parameters} />

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
