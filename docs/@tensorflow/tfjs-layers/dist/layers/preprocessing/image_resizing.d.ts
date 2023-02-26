/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/image_resizing" />
import { Rank, serialization, Tensor } from '@tensorflow/tfjs-core';
import { Layer, LayerArgs } from '../../engine/topology';
import { Shape } from '../../keras_format/common';
import { Kwargs } from '../../types';
declare const INTERPOLATION_KEYS: readonly ["bilinear", "nearest"];
declare type InterpolationType = typeof INTERPOLATION_KEYS[number];
export declare interface ResizingArgs extends LayerArgs {
    height: number;
    width: number;
    interpolation?: InterpolationType;
    cropToAspectRatio?: boolean;
}
/**
 * Preprocessing Resizing Layer
 *
 * This resizes images by a scaling and offset factor
 */
export declare class Resizing extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly height;
    private readonly width;
    private readonly interpolation;
    private readonly cropToAspectRatio;
    constructor(args: ResizingArgs);
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
    call(inputs: Tensor<Rank.R3> | Tensor<Rank.R4>, kwargs: Kwargs): Tensor[] | Tensor;
}
export {};
