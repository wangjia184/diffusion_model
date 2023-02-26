/**
 * @license
 * Copyright 2022 Google LLC.
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
/// <amd-module name="@tensorflow/tfjs-backend-cpu/dist/kernels/RaggedRange" />
import { KernelConfig, RaggedRangeInputs, TensorInfo } from '@tensorflow/tfjs-core';
import { MathBackendCPU } from '../backend_cpu';
export declare function raggedRange(args: {
    inputs: RaggedRangeInputs;
    backend: MathBackendCPU;
}): [
    TensorInfo,
    TensorInfo
];
export declare const raggedRangeConfig: KernelConfig;
