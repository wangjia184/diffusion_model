/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/center_crop" />
import { serialization, DataType, Tensor, Tensor3D, Tensor4D } from '@tensorflow/tfjs-core';
import { LayerArgs, Layer } from '../../engine/topology';
import { Kwargs } from '../../types';
import { Shape } from '../../keras_format/common';
export declare interface CenterCropArgs extends LayerArgs {
    height: number;
    width: number;
}
export declare class CenterCrop extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly height;
    private readonly width;
    constructor(args: CenterCropArgs);
    centerCrop(inputs: Tensor3D | Tensor4D, hBuffer: number, wBuffer: number, height: number, width: number, inputHeight: number, inputWidth: number, dtype: DataType): Tensor | Tensor[];
    upsize(inputs: Tensor3D | Tensor4D, height: number, width: number, dtype: DataType): Tensor | Tensor[];
    call(inputs: Tensor3D | Tensor4D, kwargs: Kwargs): Tensor[] | Tensor;
    getConfig(): serialization.ConfigDict;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
}
