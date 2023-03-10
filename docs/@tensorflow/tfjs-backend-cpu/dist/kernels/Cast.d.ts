/// <amd-module name="@tensorflow/tfjs-backend-cpu/dist/kernels/Cast" />
/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { CastAttrs, CastInputs, DataType, KernelConfig, TensorInfo, TypedArray } from '@tensorflow/tfjs-core';
import { MathBackendCPU } from '../backend_cpu';
export declare function castImpl(values: TypedArray, shape: number[], inputType: DataType, dtype: DataType): [number[], DataType, TypedArray];
export declare function cast(args: {
    inputs: CastInputs;
    backend: MathBackendCPU;
    attrs: CastAttrs;
}): TensorInfo;
export declare const castConfig: KernelConfig;
