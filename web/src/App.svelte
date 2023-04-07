<script>
  import {
    Progress,
    Button,
    TabContent,
    TabPane,
    Icon,
    Table,
  } from "sveltestrap";
  import { onMount } from "svelte";
  import ReverseProcess from "./ReverseProcess.svelte";
  import SingleStep from "./SingleStep.svelte";
  import MultiStep from "./MultiStep.svelte";

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
  let variables = null;
</script>

<svelte:head>
  <link rel="stylesheet" href="./bootstrap/dist/css/bootstrap.min.css" />
</svelte:head>

<TabContent class="w-100 mt-3 ">
  <TabPane tabId="singlestep">
    <span slot="tab">
      <Icon name="arrow-right-square-fill" /> Single Step Forward
    </span>
    <div class="border border-top-0 p-5"><SingleStep /></div></TabPane
  >
  <TabPane tabId="multistep">
    <span slot="tab">
      <Icon name="fast-forward-fill" /> Multiple Steps Forward
    </span>
    <div class="border border-top-0 p-5">
      <MultiStep />
    </div>
  </TabPane>
  <TabPane tabId="backward" active>
    <span slot="tab">
      <Icon name="arrow-left-square-fill" /> Reverse Process
    </span>
    <ReverseProcess />
  </TabPane>
</TabContent>
