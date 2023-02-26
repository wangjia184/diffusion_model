/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/preprocessing_utils" />
import { Tensor, Tensor1D, Tensor2D, TensorLike } from '@tensorflow/tfjs-core';
export declare type OutputMode = 'int' | 'oneHot' | 'multiHot' | 'count' | 'tfIdf';
export declare function encodeCategoricalInputs(inputs: Tensor | Tensor[], outputMode: OutputMode, depth: number, weights?: Tensor1D | Tensor2D | TensorLike): Tensor | Tensor[];
