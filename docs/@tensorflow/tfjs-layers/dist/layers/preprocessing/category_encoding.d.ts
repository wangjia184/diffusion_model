/**
 * @license
 * Copyright 2022 CodeSmith LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/// <amd-module name="@tensorflow/tfjs-layers/dist/layers/preprocessing/category_encoding" />
import { LayerArgs, Layer } from '../../engine/topology';
import { serialization, Tensor } from '@tensorflow/tfjs-core';
import { Shape } from '../../keras_format/common';
import { Kwargs } from '../../types';
import { OutputMode } from './preprocessing_utils';
export declare interface CategoryEncodingArgs extends LayerArgs {
    numTokens: number;
    outputMode?: OutputMode;
}
export declare class CategoryEncoding extends Layer {
    /** @nocollapse */
    static className: string;
    private readonly numTokens;
    private readonly outputMode;
    constructor(args: CategoryEncodingArgs);
    getConfig(): serialization.ConfigDict;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor[] | Tensor;
}
