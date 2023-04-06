<script>
    import {
        Table,
        Modal,
        ModalBody,
        ModalFooter,
        ModalHeader,
        Button,
    } from "sveltestrap";
    import t_jpg from "./assets/t.jpg";
    import beta_t_jpg from "./assets/beta_t.jpg";
    import alpha_t_jpg from "./assets/alpha_t.jpg";
    import alphas_cumprod_jpg from "./assets/alphas_cumprod.jpg";
    import alphas_cumprod_prev_jpg from "./assets/alphas_cumprod_prev.jpg";
    import sigma_jpg from "./assets/sigma.jpg";

    export let open = false;
    export let variables = {
        betas: [],
        alphas: [],
        alphas_cumprod: [],
        alphas_cumprod_prev: [],
        stddevs: [],
    };
</script>

<Modal isOpen={open} toggle={() => (open = false)} fullscreen={true}>
    <ModalHeader toggle={() => (open = false)}>Variables</ModalHeader>
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
            </tbody>
        </Table>
    </ModalBody>
    <ModalFooter>
        <Button color="secondary" on:click={() => (open = false)}>Close</Button>
    </ModalFooter>
</Modal>
