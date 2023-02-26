/// <amd-module name="@tensorflow/tfjs-converter/dist/executor/test_data/hash_table_v2_model_loader" />
/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
export declare const HASH_TABLE_MODEL_V2: {
    modelTopology: {
        node: ({
            name: string;
            op: string;
            attr: {
                value: {
                    tensor: {
                        dtype: string;
                        tensorShape: {};
                    };
                };
                dtype: {
                    type: string;
                };
                shape?: undefined;
                Tout?: undefined;
                Tin?: undefined;
                _has_manual_control_dependencies?: undefined;
                T?: undefined;
            };
            input?: undefined;
        } | {
            name: string;
            op: string;
            attr: {
                shape: {
                    shape: {
                        dim: {
                            size: string;
                        }[];
                    };
                };
                dtype: {
                    type: string;
                };
                value?: undefined;
                Tout?: undefined;
                Tin?: undefined;
                _has_manual_control_dependencies?: undefined;
                T?: undefined;
            };
            input?: undefined;
        } | {
            name: string;
            op: string;
            attr: {
                shape: {
                    shape: {
                        dim?: undefined;
                    };
                };
                dtype: {
                    type: string;
                };
                value?: undefined;
                Tout?: undefined;
                Tin?: undefined;
                _has_manual_control_dependencies?: undefined;
                T?: undefined;
            };
            input?: undefined;
        } | {
            name: string;
            op: string;
            input: string[];
            attr: {
                Tout: {
                    type: string;
                };
                Tin: {
                    type: string;
                };
                _has_manual_control_dependencies: {
                    b: boolean;
                };
                value?: undefined;
                dtype?: undefined;
                shape?: undefined;
                T?: undefined;
            };
        } | {
            name: string;
            op: string;
            input: string[];
            attr: {
                T: {
                    type: string;
                };
                value?: undefined;
                dtype?: undefined;
                shape?: undefined;
                Tout?: undefined;
                Tin?: undefined;
                _has_manual_control_dependencies?: undefined;
            };
        })[];
        library: {};
        versions: {
            producer: number;
        };
    };
    format: string;
    generatedBy: string;
    convertedBy: string;
    weightSpecs: {
        name: string;
        shape: number[];
        dtype: string;
    }[];
    weightData: ArrayBufferLike;
    signature: {
        inputs: {
            input: {
                name: string;
                dtype: string;
                tensorShape: {
                    dim: {
                        size: string;
                    }[];
                };
            };
            'unknown:0': {
                name: string;
                dtype: string;
                tensorShape: {};
                resourceId: number;
            };
        };
        outputs: {
            output_0: {
                name: string;
                dtype: string;
                tensorShape: {
                    dim: {
                        size: string;
                    }[];
                };
            };
        };
    };
    modelInitializer: {
        node: ({
            name: string;
            op: string;
            attr: {
                _has_manual_control_dependencies: {
                    b: boolean;
                };
                value?: undefined;
                dtype?: undefined;
                container?: undefined;
                use_node_name_sharing?: undefined;
                shared_name?: undefined;
                value_dtype?: undefined;
                key_dtype?: undefined;
                _acd_function_control_output?: undefined;
                T?: undefined;
                Tout?: undefined;
                Tin?: undefined;
            };
            input?: undefined;
        } | {
            name: string;
            op: string;
            attr: {
                value: {
                    tensor: {
                        dtype: string;
                        tensorShape: {
                            dim: {
                                size: string;
                            }[];
                        };
                    };
                };
                _has_manual_control_dependencies: {
                    b: boolean;
                };
                dtype: {
                    type: string;
                };
                container?: undefined;
                use_node_name_sharing?: undefined;
                shared_name?: undefined;
                value_dtype?: undefined;
                key_dtype?: undefined;
                _acd_function_control_output?: undefined;
                T?: undefined;
                Tout?: undefined;
                Tin?: undefined;
            };
            input?: undefined;
        } | {
            name: string;
            op: string;
            input: string[];
            attr: {
                _has_manual_control_dependencies: {
                    b: boolean;
                };
                value?: undefined;
                dtype?: undefined;
                container?: undefined;
                use_node_name_sharing?: undefined;
                shared_name?: undefined;
                value_dtype?: undefined;
                key_dtype?: undefined;
                _acd_function_control_output?: undefined;
                T?: undefined;
                Tout?: undefined;
                Tin?: undefined;
            };
        } | {
            name: string;
            op: string;
            input: string[];
            attr: {
                container: {
                    s: string;
                };
                use_node_name_sharing: {
                    b: boolean;
                };
                _has_manual_control_dependencies: {
                    b: boolean;
                };
                shared_name: {
                    s: string;
                };
                value_dtype: {
                    type: string;
                };
                key_dtype: {
                    type: string;
                };
                value?: undefined;
                dtype?: undefined;
                _acd_function_control_output?: undefined;
                T?: undefined;
                Tout?: undefined;
                Tin?: undefined;
            };
        } | {
            name: string;
            op: string;
            input: string[];
            attr: {
                _acd_function_control_output: {
                    b: boolean;
                };
                _has_manual_control_dependencies: {
                    b: boolean;
                };
                value?: undefined;
                dtype?: undefined;
                container?: undefined;
                use_node_name_sharing?: undefined;
                shared_name?: undefined;
                value_dtype?: undefined;
                key_dtype?: undefined;
                T?: undefined;
                Tout?: undefined;
                Tin?: undefined;
            };
        } | {
            name: string;
            op: string;
            input: string[];
            attr: {
                T: {
                    type: string;
                };
                _has_manual_control_dependencies?: undefined;
                value?: undefined;
                dtype?: undefined;
                container?: undefined;
                use_node_name_sharing?: undefined;
                shared_name?: undefined;
                value_dtype?: undefined;
                key_dtype?: undefined;
                _acd_function_control_output?: undefined;
                Tout?: undefined;
                Tin?: undefined;
            };
        } | {
            name: string;
            op: string;
            input: string[];
            attr: {
                T: {
                    type: string;
                };
                _has_manual_control_dependencies: {
                    b: boolean;
                };
                value?: undefined;
                dtype?: undefined;
                container?: undefined;
                use_node_name_sharing?: undefined;
                shared_name?: undefined;
                value_dtype?: undefined;
                key_dtype?: undefined;
                _acd_function_control_output?: undefined;
                Tout?: undefined;
                Tin?: undefined;
            };
        } | {
            name: string;
            op: string;
            input: string[];
            attr: {
                Tout: {
                    type: string;
                };
                Tin: {
                    type: string;
                };
                _has_manual_control_dependencies: {
                    b: boolean;
                };
                value?: undefined;
                dtype?: undefined;
                container?: undefined;
                use_node_name_sharing?: undefined;
                shared_name?: undefined;
                value_dtype?: undefined;
                key_dtype?: undefined;
                _acd_function_control_output?: undefined;
                T?: undefined;
            };
        })[];
        versions: {
            producer: number;
        };
    };
    initializerSignature: {
        outputs: {
            'Identity:0': {
                name: string;
                dtype: string;
                tensorShape: {};
                resourceId: number;
            };
        };
    };
};
