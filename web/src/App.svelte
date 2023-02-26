<script>
  import { Progress } from 'sveltestrap';

  let loadingPercentage = 0;
  let canvas;
  let idSource = 0;
  let percentage = 0;
	const commandMap = {};
  const worker = new Worker("./64x64_cosin_300/worker.js");
  worker.onmessage = async (evt) => {
		const data = evt.data;
		switch (data.type) {
			case "reply": {
				const handler = commandMap[data.id];
				if (handler) {
					handler.resolve(data.data);
				} else {
					console.log("Unable to find the promise.", data);
				}
				break;
			}
			case "ready": {
				loadingPercentage = 100;
        ddpmGenerate();
				break;
			}
			case "progress": {
				loadingPercentage = Math.floor(data.progress * 100);
				break;
			}
      case "error": {
				alert(data.message);
				break;
			}
			default: {
				console.log(data);
				break;
			}
		}
	};

  const registerPromise = () => {
		const id = idSource++;
		const promise = new Promise((resolve, reject) => {
			commandMap[id] = {
				resolve: resolve,
				reject: reject,
			};
		});
		return { id: id, promise: promise };
	};

  const ddpmStart = async () => {
		const { id, promise } = registerPromise();
		const msg = {
			type: "ddpmStart",
			id: id,
			kwargs: {
			},
		};
		worker.postMessage(msg);
		return await promise;
	};

  const ddpmNext = async (key) => {
		const { id, promise } = registerPromise();
		const msg = {
			type: "ddpmNext",
			id: id,
			kwargs: {
        key : key
			},
		};
		worker.postMessage(msg);
		return await promise;
	};

  const ddpmGenerate = async() => {
    let response = await ddpmStart();
    updateCanvas(response.image);
    percentage = response.percent;

    while(response.step > 0){
      response = await ddpmNext(response.key);
      updateCanvas(response.image);
      percentage = response.percent;
    }
  };

  const updateCanvas = (img) => {
    const height = img[0].length;
    const width = img[0][0].length;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const id = ctx.createImageData(width, height, { colorSpace : 'srgb'});
    const pixels = id.data;

    for( let y = 0; y < height; y ++){
        for( let x = 0; x < width; x++) {
            const offset = (y * width + x) * 4;
            const channels = img[0][y][x];

            pixels[offset] = Math.max( 0, Math.min( 255, channels[0] * 127.5 + 127.5));
            pixels[offset + 1] = Math.max( 0, Math.min( 255, channels[1] * 127.5 + 127.5));
            pixels[offset + 2] = Math.max( 0, Math.min( 255, channels[2] * 127.5 + 127.5));
            pixels[offset + 3] = 255;
        }
    }                  

    ctx.putImageData(id, 0, 0);
    /*
    const image = new Image();
    image.src = canvas.toDataURL();
    while (container.firstChild) {
        container.removeChild(container.lastChild);
    }
    container.appendChild(image);
    */
  }
</script>

<svelte:head>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
</svelte:head>

{#if loadingPercentage < 100 }
<div>
  <div class="text-center">Loading ...</div>
  <Progress animated color="success" value={loadingPercentage}>{loadingPercentage}%</Progress>
</div>
{:else}
<div class="d-flex justify-content-center">
  <div >
    <canvas class="d-flex" bind:this={canvas}></canvas>
    <div class="d-flex progress_bar" style="width:{percentage*100}%"></div>
  </div>
</div>
{/if}


<main>
  
</main>


<style>
  .progress_bar {
    height: 3px;
    background-color: darkorange;
  }

  :global(canvas) {
    width : 64px;
    height: 64px;
    background-color: #999;
  }
</style>