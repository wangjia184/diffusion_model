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
import { util } from '@tensorflow/tfjs-core';
import { Im2ColPackedProgram } from '../im2col_packed_gpu';
import { mapActivationToShaderProgram } from '../kernel_utils/kernel_funcs_utils';
import { MatMulPackedProgram } from '../mulmat_packed_gpu';
import * as webgl_util from '../webgl_util';
import { batchMatMulImpl, MATMUL_SHARED_DIM_THRESHOLD } from './BatchMatMul_impl';
import { identity } from './Identity';
import { reshape } from './Reshape';
// Both conv2dByMatMul and conv2dWithIm2Row fuse height and width into one
// dimension to compute batchMatMul, so bias and activation weights are also
// supposed to fuse the two dimensions into one.
//
// This function computes the target shape for fusing height and width
// dimensions. Returning null means the shape is already compatible.
//
// Even though the bias is not supposed to be a 3-D or a 4-D (including
// batch) tensor and PReLU activiation weights is not supposed to be a 4-D
// tensor, we still need to support them, because we haven't disabled
// them for NHWC format.
// https://github.com/tensorflow/tfjs/blob/b53bd47e880367ae57493f0ea628abaf08db2d5d/tfjs-core/src/ops/fused/conv2d.ts#L181-L196
function getShapeForBatchMatMul(shape, isChannelsLast) {
    const length = shape.length;
    if (length >= 3) {
        return isChannelsLast ?
            [
                ...shape.slice(0, -3) /* batch */,
                shape[length - 3] * shape[length - 2] /* height * width */,
                shape[length - 1] /* channel */
            ] :
            [
                ...shape.slice(0, -3) /* batch */, shape[length - 3] /* channel */,
                shape[length - 2] * shape[length - 1] /* height * width */
            ];
    }
    else if (!isChannelsLast && length === 1 && shape[0] > 1) {
        return [shape[0], 1];
    }
    else {
        return null;
    }
}
// For 1x1 kernels that iterate through every point in the input, convolution
// can be expressed as matrix multiplication (without need for memory
// remapping).
export function conv2dByMatMul({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    // Reshapes conv2D input to 2D tensors, uses matMul and then reshape the
    // result from 2D to 4D.
    const xShape = x.shape;
    const xTexData = backend.texData.get(x.dataId);
    const sharedMatMulDim = convInfo.inChannels;
    const outerShapeX = xShape[0] * xShape[1] * xShape[2];
    const outerShapeFilter = convInfo.outChannels;
    const isChannelsLast = convInfo.dataFormat === 'channelsLast';
    const transposeA = false;
    const transposeB = false;
    let out;
    const intermediates = [];
    if (preluActivationWeights != null) {
        const targetShape = getShapeForBatchMatMul(preluActivationWeights.shape, isChannelsLast);
        if (targetShape != null) {
            preluActivationWeights = reshape({
                inputs: { x: preluActivationWeights },
                backend,
                attrs: { shape: targetShape }
            });
            intermediates.push(preluActivationWeights);
        }
    }
    if (bias != null) {
        const targetShape = getShapeForBatchMatMul(bias.shape, isChannelsLast);
        if (targetShape != null) {
            bias = reshape({ inputs: { x: bias }, backend, attrs: { shape: targetShape } });
            intermediates.push(bias);
        }
    }
    // TODO: Once reduction ops are packed, batchMatMul will always be packed
    // and we can remove this condition.
    const batchMatMulWillBeUnpacked = (outerShapeX === 1 || outerShapeFilter === 1) &&
        sharedMatMulDim > MATMUL_SHARED_DIM_THRESHOLD;
    // The algorithm in the if condition assumes (1) the output will be packed,
    // (2) x is packed, (3) x isChannelsLast, (4)  x's packed texture is already
    // on GPU, (5) col is odd, (6) the width, height and inChannels are the same
    // for xTexData.shape and xShape.
    const canOptimize = !batchMatMulWillBeUnpacked && xTexData.isPacked &&
        isChannelsLast && xTexData.texture != null && xShape[2] % 2 !== 0 &&
        util.arraysEqual(xTexData.shape.slice(-3), xShape.slice(-3));
    if (canOptimize) {
        // We avoid expensive packed 2x2 reshape by padding col count to next,
        // even number. When col is odd, the result of packed batchMatMul is
        // the same (has the same texture layout and and values in the texture) as
        // it is for next even col. We make the odd-cols tensor to look like
        // even-cols tensor before the operation and, after the batchMatMul,
        // fix the even-cols result to have odd number of cols.
        const targetShape = xShape[0] * xShape[1] * (xShape[2] + 1);
        const xReshaped = {
            dataId: x.dataId,
            shape: [1, targetShape, convInfo.inChannels],
            dtype: x.dtype
        };
        // xTexData.shape gets referenced from GPGPUBinary.inShapeInfos.
        // Decrementing col count, after batchMatMul->...->compileProgram leads to
        // invalid col count within the reference in GPGPUBinary.inShapeInfos.
        // Alternative fix would be to provide a copy to GPGPUBinary.inShapeInfos
        // in compileProgram method, but that would affect compilation of all
        // programs - instead, provide a copy here, with even col count, before
        // calling batchMatMul->...->compileProgram and after that, the original
        // xTexData.shape is restored.
        const originalXTexDataShape = xTexData.shape;
        xTexData.shape = xTexData.shape.slice();
        xTexData.shape[xTexData.shape.length - 2]++;
        util.assert(webgl_util.isReshapeFree(xTexData.shape, xReshaped.shape), () => `packed reshape ${xTexData.shape} to ${xReshaped.shape} isn't free`);
        const filterReshaped = reshape({
            inputs: { x: filter },
            backend,
            attrs: { shape: [1, convInfo.inChannels, convInfo.outChannels] }
        });
        intermediates.push(filterReshaped);
        const pointwiseConv = batchMatMulImpl({
            a: xReshaped,
            b: filterReshaped,
            backend,
            transposeA,
            transposeB,
            bias,
            activation,
            preluActivationWeights,
            leakyreluAlpha
        });
        const pointwiseConvTexData = backend.texData.get(pointwiseConv.dataId);
        util.assert(pointwiseConvTexData.isPacked, () => 'batchMatMul result is expected to be packed');
        // Restore the input shape to original.
        xTexData.shape = originalXTexDataShape;
        // Set the output shape - there is no need for expensive reshape as data
        // layout is already correct.
        pointwiseConvTexData.shape = convInfo.outShape;
        out = identity({ inputs: { x: pointwiseConv }, backend });
        out.shape = convInfo.outShape;
        intermediates.push(pointwiseConv);
    }
    else {
        const numCols = convInfo.outHeight * convInfo.outWidth;
        const xReshaped = reshape({
            inputs: { x },
            backend,
            attrs: {
                shape: isChannelsLast ?
                    [convInfo.batchSize, numCols, convInfo.inChannels] :
                    [convInfo.batchSize, convInfo.inChannels, numCols]
            }
        });
        const filterReshaped = reshape({
            inputs: { x: filter },
            backend,
            attrs: { shape: [1, convInfo.inChannels, convInfo.outChannels] }
        });
        const result = batchMatMulImpl({
            a: isChannelsLast ? xReshaped : filterReshaped,
            b: isChannelsLast ? filterReshaped : xReshaped,
            transposeA: !isChannelsLast,
            transposeB,
            backend,
            bias,
            activation,
            preluActivationWeights,
            leakyreluAlpha
        });
        out = reshape({ inputs: { x: result }, backend, attrs: { shape: convInfo.outShape } });
        intermediates.push(xReshaped);
        intermediates.push(filterReshaped);
        intermediates.push(result);
    }
    for (const i of intermediates) {
        backend.disposeIntermediateTensorInfo(i);
    }
    return out;
}
// Implements the im2row algorithm as outlined in "High Performance
// Convolutional Neural Networks for Document Processing" (Suvisoft, 2006)
export function conv2dWithIm2Row({ x, filter, convInfo, backend, bias = null, preluActivationWeights = null, leakyreluAlpha = 0, activation = null }) {
    // Rearranges conv2d input so each block to be convolved over forms the
    // column of a new matrix with shape [filterWidth * filterHeight *
    // inChannels, outHeight * outWidth]. The filter is also rearranged so each
    // output channel forms a row of a new matrix with shape [outChannels,
    // filterWidth * filterHeight * inChannels]. The convolution is then
    // computed by multiplying these matrices and reshaping the result.
    const { filterWidth, filterHeight, inChannels, outWidth, outHeight, dataFormat } = convInfo;
    const isChannelsLast = dataFormat === 'channelsLast';
    const sharedDim = filterWidth * filterHeight * inChannels;
    const numCols = outHeight * outWidth;
    const x2ColShape = [convInfo.batchSize, sharedDim, numCols];
    const transposeA = true;
    const transposeB = false;
    const intermediates = [];
    if (preluActivationWeights != null) {
        const targetShape = getShapeForBatchMatMul(preluActivationWeights.shape, isChannelsLast);
        if (targetShape != null) {
            preluActivationWeights = reshape({
                inputs: { x: preluActivationWeights },
                backend,
                attrs: { shape: targetShape }
            });
            intermediates.push(preluActivationWeights);
        }
    }
    if (bias != null) {
        const targetShape = getShapeForBatchMatMul(bias.shape, isChannelsLast);
        if (targetShape != null) {
            bias = reshape({ inputs: { x: bias }, backend, attrs: { shape: targetShape } });
            intermediates.push(bias);
        }
    }
    const w2Row = reshape({
        inputs: { x: filter },
        backend,
        attrs: { shape: [1, sharedDim, util.sizeFromShape(filter.shape) / sharedDim] }
    });
    intermediates.push(w2Row);
    const im2ColProgram = new Im2ColPackedProgram(x2ColShape, convInfo);
    const customValues = [
        x.shape, [convInfo.padInfo.top, convInfo.padInfo.left],
        [convInfo.strideHeight, convInfo.strideWidth],
        [convInfo.dilationHeight, convInfo.dilationWidth], [convInfo.inChannels],
        [convInfo.filterWidth * convInfo.inChannels], [convInfo.outWidth]
    ];
    const im2Col = backend.runWebGLProgram(im2ColProgram, [x], 'float32', customValues);
    const im2ColReshaped = reshape({ inputs: { x: im2Col }, backend, attrs: { shape: x2ColShape } });
    intermediates.push(im2Col);
    intermediates.push(im2ColReshaped);
    const hasBias = bias != null;
    const hasPreluActivationWeights = preluActivationWeights != null;
    const hasLeakyreluAlpha = activation === 'leakyrelu';
    const fusedActivation = activation ? mapActivationToShaderProgram(activation, true) : null;
    const matmulProgram = new MatMulPackedProgram(isChannelsLast ? im2ColReshaped.shape :
        w2Row.shape, isChannelsLast ? w2Row.shape :
        im2ColReshaped.shape, isChannelsLast ? [convInfo.batchSize, numCols, convInfo.outChannels] :
        [convInfo.batchSize, convInfo.outChannels, numCols], transposeA, transposeB, hasBias, fusedActivation, hasPreluActivationWeights, hasLeakyreluAlpha);
    const inputs = isChannelsLast ? [im2ColReshaped, w2Row] : [w2Row, im2ColReshaped];
    if (bias) {
        inputs.push(bias);
    }
    if (hasPreluActivationWeights) {
        inputs.push(preluActivationWeights);
    }
    if (hasLeakyreluAlpha) {
        const $leakyreluAlpha = backend.makeTensorInfo([], 'float32', util.createScalarValue(leakyreluAlpha, 'float32'));
        inputs.push($leakyreluAlpha);
        intermediates.push($leakyreluAlpha);
    }
    const product = backend.runWebGLProgram(matmulProgram, inputs, 'float32');
    const out = reshape({ inputs: { x: product }, backend, attrs: { shape: convInfo.outShape } });
    intermediates.push(product);
    for (const i of intermediates) {
        backend.disposeIntermediateTensorInfo(i);
    }
    return out;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ29udjJEX2ltcGwuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWJhY2tlbmQtd2ViZ2wvc3JjL2tlcm5lbHMvQ29udjJEX2ltcGwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUEyQixJQUFJLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUtyRSxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUN6RCxPQUFPLEVBQUMsNEJBQTRCLEVBQUMsTUFBTSxvQ0FBb0MsQ0FBQztBQUNoRixPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUN6RCxPQUFPLEtBQUssVUFBVSxNQUFNLGVBQWUsQ0FBQztBQUU1QyxPQUFPLEVBQUMsZUFBZSxFQUFFLDJCQUEyQixFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDaEYsT0FBTyxFQUFDLFFBQVEsRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNwQyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBYWxDLDBFQUEwRTtBQUMxRSw0RUFBNEU7QUFDNUUsZ0RBQWdEO0FBQ2hELEVBQUU7QUFDRixzRUFBc0U7QUFDdEUsb0VBQW9FO0FBQ3BFLEVBQUU7QUFDRix1RUFBdUU7QUFDdkUsMEVBQTBFO0FBQzFFLHFFQUFxRTtBQUNyRSx3QkFBd0I7QUFDeEIsK0hBQStIO0FBQy9ILFNBQVMsc0JBQXNCLENBQzNCLEtBQWUsRUFBRSxjQUF1QjtJQUMxQyxNQUFNLE1BQU0sR0FBRyxLQUFLLENBQUMsTUFBTSxDQUFDO0lBQzVCLElBQUksTUFBTSxJQUFJLENBQUMsRUFBRTtRQUNmLE9BQU8sY0FBYyxDQUFDLENBQUM7WUFDbkI7Z0JBQ0UsR0FBRyxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLFdBQVc7Z0JBQ2pDLEtBQUss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QUFDLEtBQUssRUFBRSxjQUFjLENBQUMsQ0FBQztJQUN2RSxJQUFJLElBQUksRUFBRTtRQUNSLE1BQU0sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUM7S0FDbkI7SUFDRCxJQUFJLHlCQUF5QixFQUFFO1FBQzdCLE1BQU0sQ0FBQyxJQUFJLENBQUMsc0JBQXNCLENBQUMsQ0FBQztLQUNyQztJQUNELElBQUksaUJBQWlCLEVBQUU7UUFDckIsTUFBTSxlQUFlLEdBQUcsT0FBTyxDQUFDLGNBQWMsQ0FDMUMsRUFBRSxFQUFFLFNBQVMsRUFDYixJQUFJLENBQUMsaUJBQWlCLENBQUMsY0FBc0MsRUFDdEMsU0FBUyxDQUFDLENBQUMsQ0FBQztRQUN2QyxNQUFNLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxDQUFDO1FBQzdCLGFBQWEsQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLENBQUM7S0FDckM7SUFDRCxNQUFNLE9BQU8sR0FBRyxPQUFPLENBQUMsZUFBZSxDQUFDLGFBQWEsRUFBRSxNQUFNLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDMUUsTUFBTSxHQUFHLEdBQUcsT0FBTyxDQUNmLEVBQUMsTUFBTSxFQUFFLEVBQUMsQ0FBQyxFQUFFLE9BQU8sRUFBQyxFQUFFLE9BQU8sRUFBRSxLQUFLLEVBQUUsRUFBQyxLQUFLLEVBQUUsUUFBUSxDQUFDLFFBQVEsRUFBQyxFQUFDLENBQUMsQ0FBQztJQUV4RSxhQUFhLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzVCLEtBQUssTUFBTSxDQUFDLElBQUksYUFBYSxFQUFFO1FBQzdCLE9BQU8sQ0FBQyw2QkFBNkIsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUMxQztJQUVELE9BQU8sR0FBRyxDQUFDO0FBQ2IsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtiYWNrZW5kX3V0aWwsIFRlbnNvckluZm8sIHV0aWx9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbi8vIGltcG9ydCB7YXNzZXJ0QW5kR2V0QnJvYWRjYXN0U2hhcGV9IGZyb21cbi8vICcuLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9icm9hZGNhc3RfdXRpbCc7XG5pbXBvcnQge01hdGhCYWNrZW5kV2ViR0x9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ2wnO1xuaW1wb3J0IHtJbTJDb2xQYWNrZWRQcm9ncmFtfSBmcm9tICcuLi9pbTJjb2xfcGFja2VkX2dwdSc7XG5pbXBvcnQge21hcEFjdGl2YXRpb25Ub1NoYWRlclByb2dyYW19IGZyb20gJy4uL2tlcm5lbF91dGlscy9rZXJuZWxfZnVuY3NfdXRpbHMnO1xuaW1wb3J0IHtNYXRNdWxQYWNrZWRQcm9ncmFtfSBmcm9tICcuLi9tdWxtYXRfcGFja2VkX2dwdSc7XG5pbXBvcnQgKiBhcyB3ZWJnbF91dGlsIGZyb20gJy4uL3dlYmdsX3V0aWwnO1xuXG5pbXBvcnQge2JhdGNoTWF0TXVsSW1wbCwgTUFUTVVMX1NIQVJFRF9ESU1fVEhSRVNIT0xEfSBmcm9tICcuL0JhdGNoTWF0TXVsX2ltcGwnO1xuaW1wb3J0IHtpZGVudGl0eX0gZnJvbSAnLi9JZGVudGl0eSc7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vUmVzaGFwZSc7XG5cbnR5cGUgQ29udjJEQ29uZmlnID0ge1xuICB4OiBUZW5zb3JJbmZvLFxuICBmaWx0ZXI6IFRlbnNvckluZm8sXG4gIGNvbnZJbmZvOiBiYWNrZW5kX3V0aWwuQ29udjJESW5mbyxcbiAgYmFja2VuZDogTWF0aEJhY2tlbmRXZWJHTCxcbiAgYmlhcz86IFRlbnNvckluZm8sXG4gIHByZWx1QWN0aXZhdGlvbldlaWdodHM/OiBUZW5zb3JJbmZvLFxuICBsZWFreXJlbHVBbHBoYT86IG51bWJlcixcbiAgYWN0aXZhdGlvbj86IGJhY2tlbmRfdXRpbC5BY3RpdmF0aW9uXG59O1xuXG4vLyBCb3RoIGNvbnYyZEJ5TWF0TXVsIGFuZCBjb252MmRXaXRoSW0yUm93IGZ1c2UgaGVpZ2h0IGFuZCB3aWR0aCBpbnRvIG9uZVxuLy8gZGltZW5zaW9uIHRvIGNvbXB1dGUgYmF0Y2hNYXRNdWwsIHNvIGJpYXMgYW5kIGFjdGl2YXRpb24gd2VpZ2h0cyBhcmUgYWxzb1xuLy8gc3VwcG9zZWQgdG8gZnVzZSB0aGUgdHdvIGRpbWVuc2lvbnMgaW50byBvbmUuXG4vL1xuLy8gVGhpcyBmdW5jdGlvbiBjb21wdXRlcyB0aGUgdGFyZ2V0IHNoYXBlIGZvciBmdXNpbmcgaGVpZ2h0IGFuZCB3aWR0aFxuLy8gZGltZW5zaW9ucy4gUmV0dXJuaW5nIG51bGwgbWVhbnMgdGhlIHNoYXBlIGlzIGFscmVhZHkgY29tcGF0aWJsZS5cbi8vXG4vLyBFdmVuIHRob3VnaCB0aGUgYmlhcyBpcyBub3Qgc3VwcG9zZWQgdG8gYmUgYSAzLUQgb3IgYSA0LUQgKGluY2x1ZGluZ1xuLy8gYmF0Y2gpIHRlbnNvciBhbmQgUFJlTFUgYWN0aXZpYXRpb24gd2VpZ2h0cyBpcyBub3Qgc3VwcG9zZWQgdG8gYmUgYSA0LURcbi8vIHRlbnNvciwgd2Ugc3RpbGwgbmVlZCB0byBzdXBwb3J0IHRoZW0sIGJlY2F1c2Ugd2UgaGF2ZW4ndCBkaXNhYmxlZFxuLy8gdGhlbSBmb3IgTkhXQyBmb3JtYXQuXG4vLyBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL2Jsb2IvYjUzYmQ0N2U4ODAzNjdhZTU3NDkzZjBlYTYyOGFiYWYwOGRiMmQ1ZC90ZmpzLWNvcmUvc3JjL29wcy9mdXNlZC9jb252MmQudHMjTDE4MS1MMTk2XG5mdW5jdGlvbiBnZXRTaGFwZUZvckJhdGNoTWF0TXVsKFxuICAgIHNoYXBlOiBudW1iZXJbXSwgaXNDaGFubmVsc0xhc3Q6IGJvb2xlYW4pOiBudW1iZXJbXSB7XG4gIGNvbnN0IGxlbmd0aCA9IHNoYXBlLmxlbmd0aDtcbiAgaWYgKGxlbmd0aCA+PSAzKSB7XG4gICAgcmV0dXJuIGlzQ2hhbm5lbHNMYXN0ID9cbiAgICAgICAgW1xuICAgICAgICAgIC4uLnNoYXBlLnNsaWNlKDAsIC0zKSAvKiBiYXRjaCAqLyxcbiAgICAgICAgICBzaGFwZVtsZW5ndGggLSAzXSAqIHNoYXBlW2xlbmd0aCAtIDJdIC8qIGhlaWdodCAqIHdpZHRoICovLFxuICAgICAgICAgIHNoYXBlW2xlbmd0aCAtIDFdIC8qIGNoYW5uZWwgKi9cbiAgICAgICAgXSA6XG4gICAgICAgIFtcbiAgICAgICAgICAuLi5zaGFwZS5zbGljZSgwLCAtMykgLyogYmF0Y2ggKi8sIHNoYXBlW2xlbmd0aCAtIDNdIC8qIGNoYW5uZWwgKi8sXG4gICAgICAgICAgc2hhcGVbbGVuZ3RoIC0gMl0gKiBzaGFwZVtsZW5ndGggLSAxXSAvKiBoZWlnaHQgKiB3aWR0aCAqL1xuICAgICAgICBdO1xuICB9IGVsc2UgaWYgKCFpc0NoYW5uZWxzTGFzdCAmJiBsZW5ndGggPT09IDEgJiYgc2hhcGVbMF0gPiAxKSB7XG4gICAgcmV0dXJuIFtzaGFwZVswXSwgMV07XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIG51bGw7XG4gIH1cbn1cblxuLy8gRm9yIDF4MSBrZXJuZWxzIHRoYXQgaXRlcmF0ZSB0aHJvdWdoIGV2ZXJ5IHBvaW50IGluIHRoZSBpbnB1dCwgY29udm9sdXRpb25cbi8vIGNhbiBiZSBleHByZXNzZWQgYXMgbWF0cml4IG11bHRpcGxpY2F0aW9uICh3aXRob3V0IG5lZWQgZm9yIG1lbW9yeVxuLy8gcmVtYXBwaW5nKS5cbmV4cG9ydCBmdW5jdGlvbiBjb252MmRCeU1hdE11bCh7XG4gIHgsXG4gIGZpbHRlcixcbiAgY29udkluZm8sXG4gIGJhY2tlbmQsXG4gIGJpYXMgPSBudWxsLFxuICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gbnVsbCxcbiAgbGVha3lyZWx1QWxwaGEgPSAwLFxuICBhY3RpdmF0aW9uID0gbnVsbFxufTogQ29udjJEQ29uZmlnKSB7XG4gIC8vIFJlc2hhcGVzIGNvbnYyRCBpbnB1dCB0byAyRCB0ZW5zb3JzLCB1c2VzIG1hdE11bCBhbmQgdGhlbiByZXNoYXBlIHRoZVxuICAvLyByZXN1bHQgZnJvbSAyRCB0byA0RC5cbiAgY29uc3QgeFNoYXBlID0geC5zaGFwZTtcbiAgY29uc3QgeFRleERhdGEgPSBiYWNrZW5kLnRleERhdGEuZ2V0KHguZGF0YUlkKTtcbiAgY29uc3Qgc2hhcmVkTWF0TXVsRGltID0gY29udkluZm8uaW5DaGFubmVscztcbiAgY29uc3Qgb3V0ZXJTaGFwZVggPSB4U2hhcGVbMF0gKiB4U2hhcGVbMV0gKiB4U2hhcGVbMl07XG4gIGNvbnN0IG91dGVyU2hhcGVGaWx0ZXIgPSBjb252SW5mby5vdXRDaGFubmVscztcbiAgY29uc3QgaXNDaGFubmVsc0xhc3QgPSBjb252SW5mby5kYXRhRm9ybWF0ID09PSAnY2hhbm5lbHNMYXN0JztcbiAgY29uc3QgdHJhbnNwb3NlQSA9IGZhbHNlO1xuICBjb25zdCB0cmFuc3Bvc2VCID0gZmFsc2U7XG5cbiAgbGV0IG91dDogVGVuc29ySW5mbztcbiAgY29uc3QgaW50ZXJtZWRpYXRlczogVGVuc29ySW5mb1tdID0gW107XG5cbiAgaWYgKHByZWx1QWN0aXZhdGlvbldlaWdodHMgIT0gbnVsbCkge1xuICAgIGNvbnN0IHRhcmdldFNoYXBlID1cbiAgICAgICAgZ2V0U2hhcGVGb3JCYXRjaE1hdE11bChwcmVsdUFjdGl2YXRpb25XZWlnaHRzLnNoYXBlLCBpc0NoYW5uZWxzTGFzdCk7XG4gICAgaWYgKHRhcmdldFNoYXBlICE9IG51bGwpIHtcbiAgICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHMgPSByZXNoYXBlKHtcbiAgICAgICAgaW5wdXRzOiB7eDogcHJlbHVBY3RpdmF0aW9uV2VpZ2h0c30sXG4gICAgICAgIGJhY2tlbmQsXG4gICAgICAgIGF0dHJzOiB7c2hhcGU6IHRhcmdldFNoYXBlfVxuICAgICAgfSk7XG4gICAgICBpbnRlcm1lZGlhdGVzLnB1c2gocHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyk7XG4gICAgfVxuICB9XG5cbiAgaWYgKGJpYXMgIT0gbnVsbCkge1xuICAgIGNvbnN0IHRhcmdldFNoYXBlID0gZ2V0U2hhcGVGb3JCYXRjaE1hdE11bChiaWFzLnNoYXBlLCBpc0NoYW5uZWxzTGFzdCk7XG4gICAgaWYgKHRhcmdldFNoYXBlICE9IG51bGwpIHtcbiAgICAgIGJpYXMgPSByZXNoYXBlKHtpbnB1dHM6IHt4OiBiaWFzfSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogdGFyZ2V0U2hhcGV9fSk7XG4gICAgICBpbnRlcm1lZGlhdGVzLnB1c2goYmlhcyk7XG4gICAgfVxuICB9XG5cbiAgLy8gVE9ETzogT25jZSByZWR1Y3Rpb24gb3BzIGFyZSBwYWNrZWQsIGJhdGNoTWF0TXVsIHdpbGwgYWx3YXlzIGJlIHBhY2tlZFxuICAvLyBhbmQgd2UgY2FuIHJlbW92ZSB0aGlzIGNvbmRpdGlvbi5cbiAgY29uc3QgYmF0Y2hNYXRNdWxXaWxsQmVVbnBhY2tlZCA9XG4gICAgICAob3V0ZXJTaGFwZVggPT09IDEgfHwgb3V0ZXJTaGFwZUZpbHRlciA9PT0gMSkgJiZcbiAgICAgIHNoYXJlZE1hdE11bERpbSA+IE1BVE1VTF9TSEFSRURfRElNX1RIUkVTSE9MRDtcblxuICAvLyBUaGUgYWxnb3JpdGhtIGluIHRoZSBpZiBjb25kaXRpb24gYXNzdW1lcyAoMSkgdGhlIG91dHB1dCB3aWxsIGJlIHBhY2tlZCxcbiAgLy8gKDIpIHggaXMgcGFja2VkLCAoMykgeCBpc0NoYW5uZWxzTGFzdCwgKDQpICB4J3MgcGFja2VkIHRleHR1cmUgaXMgYWxyZWFkeVxuICAvLyBvbiBHUFUsICg1KSBjb2wgaXMgb2RkLCAoNikgdGhlIHdpZHRoLCBoZWlnaHQgYW5kIGluQ2hhbm5lbHMgYXJlIHRoZSBzYW1lXG4gIC8vIGZvciB4VGV4RGF0YS5zaGFwZSBhbmQgeFNoYXBlLlxuICBjb25zdCBjYW5PcHRpbWl6ZSA9ICFiYXRjaE1hdE11bFdpbGxCZVVucGFja2VkICYmIHhUZXhEYXRhLmlzUGFja2VkICYmXG4gICAgICBpc0NoYW5uZWxzTGFzdCAmJiB4VGV4RGF0YS50ZXh0dXJlICE9IG51bGwgJiYgeFNoYXBlWzJdICUgMiAhPT0gMCAmJlxuICAgICAgdXRpbC5hcnJheXNFcXVhbCh4VGV4RGF0YS5zaGFwZS5zbGljZSgtMyksIHhTaGFwZS5zbGljZSgtMykpO1xuXG4gIGlmIChjYW5PcHRpbWl6ZSkge1xuICAgIC8vIFdlIGF2b2lkIGV4cGVuc2l2ZSBwYWNrZWQgMngyIHJlc2hhcGUgYnkgcGFkZGluZyBjb2wgY291bnQgdG8gbmV4dCxcbiAgICAvLyBldmVuIG51bWJlci4gV2hlbiBjb2wgaXMgb2RkLCB0aGUgcmVzdWx0IG9mIHBhY2tlZCBiYXRjaE1hdE11bCBpc1xuICAgIC8vIHRoZSBzYW1lIChoYXMgdGhlIHNhbWUgdGV4dHVyZSBsYXlvdXQgYW5kIGFuZCB2YWx1ZXMgaW4gdGhlIHRleHR1cmUpIGFzXG4gICAgLy8gaXQgaXMgZm9yIG5leHQgZXZlbiBjb2wuIFdlIG1ha2UgdGhlIG9kZC1jb2xzIHRlbnNvciB0byBsb29rIGxpa2VcbiAgICAvLyBldmVuLWNvbHMgdGVuc29yIGJlZm9yZSB0aGUgb3BlcmF0aW9uIGFuZCwgYWZ0ZXIgdGhlIGJhdGNoTWF0TXVsLFxuICAgIC8vIGZpeCB0aGUgZXZlbi1jb2xzIHJlc3VsdCB0byBoYXZlIG9kZCBudW1iZXIgb2YgY29scy5cbiAgICBjb25zdCB0YXJnZXRTaGFwZSA9IHhTaGFwZVswXSAqIHhTaGFwZVsxXSAqICh4U2hhcGVbMl0gKyAxKTtcbiAgICBjb25zdCB4UmVzaGFwZWQ6IFRlbnNvckluZm8gPSB7XG4gICAgICBkYXRhSWQ6IHguZGF0YUlkLFxuICAgICAgc2hhcGU6IFsxLCB0YXJnZXRTaGFwZSwgY29udkluZm8uaW5DaGFubmVsc10sXG4gICAgICBkdHlwZTogeC5kdHlwZVxuICAgIH07XG4gICAgLy8geFRleERhdGEuc2hhcGUgZ2V0cyByZWZlcmVuY2VkIGZyb20gR1BHUFVCaW5hcnkuaW5TaGFwZUluZm9zLlxuICAgIC8vIERlY3JlbWVudGluZyBjb2wgY291bnQsIGFmdGVyIGJhdGNoTWF0TXVsLT4uLi4tPmNvbXBpbGVQcm9ncmFtIGxlYWRzIHRvXG4gICAgLy8gaW52YWxpZCBjb2wgY291bnQgd2l0aGluIHRoZSByZWZlcmVuY2UgaW4gR1BHUFVCaW5hcnkuaW5TaGFwZUluZm9zLlxuICAgIC8vIEFsdGVybmF0aXZlIGZpeCB3b3VsZCBiZSB0byBwcm92aWRlIGEgY29weSB0byBHUEdQVUJpbmFyeS5pblNoYXBlSW5mb3NcbiAgICAvLyBpbiBjb21waWxlUHJvZ3JhbSBtZXRob2QsIGJ1dCB0aGF0IHdvdWxkIGFmZmVjdCBjb21waWxhdGlvbiBvZiBhbGxcbiAgICAvLyBwcm9ncmFtcyAtIGluc3RlYWQsIHByb3ZpZGUgYSBjb3B5IGhlcmUsIHdpdGggZXZlbiBjb2wgY291bnQsIGJlZm9yZVxuICAgIC8vIGNhbGxpbmcgYmF0Y2hNYXRNdWwtPi4uLi0+Y29tcGlsZVByb2dyYW0gYW5kIGFmdGVyIHRoYXQsIHRoZSBvcmlnaW5hbFxuICAgIC8vIHhUZXhEYXRhLnNoYXBlIGlzIHJlc3RvcmVkLlxuICAgIGNvbnN0IG9yaWdpbmFsWFRleERhdGFTaGFwZSA9IHhUZXhEYXRhLnNoYXBlO1xuICAgIHhUZXhEYXRhLnNoYXBlID0geFRleERhdGEuc2hhcGUuc2xpY2UoKTtcbiAgICB4VGV4RGF0YS5zaGFwZVt4VGV4RGF0YS5zaGFwZS5sZW5ndGggLSAyXSsrO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB3ZWJnbF91dGlsLmlzUmVzaGFwZUZyZWUoeFRleERhdGEuc2hhcGUsIHhSZXNoYXBlZC5zaGFwZSksXG4gICAgICAgICgpID0+IGBwYWNrZWQgcmVzaGFwZSAke3hUZXhEYXRhLnNoYXBlfSB0byAke1xuICAgICAgICAgICAgeFJlc2hhcGVkLnNoYXBlfSBpc24ndCBmcmVlYCk7XG4gICAgY29uc3QgZmlsdGVyUmVzaGFwZWQgPSByZXNoYXBlKHtcbiAgICAgIGlucHV0czoge3g6IGZpbHRlcn0sXG4gICAgICBiYWNrZW5kLFxuICAgICAgYXR0cnM6IHtzaGFwZTogWzEsIGNvbnZJbmZvLmluQ2hhbm5lbHMsIGNvbnZJbmZvLm91dENoYW5uZWxzXX1cbiAgICB9KTtcbiAgICBpbnRlcm1lZGlhdGVzLnB1c2goZmlsdGVyUmVzaGFwZWQpO1xuICAgIGNvbnN0IHBvaW50d2lzZUNvbnYgPSBiYXRjaE1hdE11bEltcGwoe1xuICAgICAgYTogeFJlc2hhcGVkLFxuICAgICAgYjogZmlsdGVyUmVzaGFwZWQsXG4gICAgICBiYWNrZW5kLFxuICAgICAgdHJhbnNwb3NlQSxcbiAgICAgIHRyYW5zcG9zZUIsXG4gICAgICBiaWFzLFxuICAgICAgYWN0aXZhdGlvbixcbiAgICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHMsXG4gICAgICBsZWFreXJlbHVBbHBoYVxuICAgIH0pO1xuXG4gICAgY29uc3QgcG9pbnR3aXNlQ29udlRleERhdGEgPSBiYWNrZW5kLnRleERhdGEuZ2V0KHBvaW50d2lzZUNvbnYuZGF0YUlkKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgcG9pbnR3aXNlQ29udlRleERhdGEuaXNQYWNrZWQsXG4gICAgICAgICgpID0+ICdiYXRjaE1hdE11bCByZXN1bHQgaXMgZXhwZWN0ZWQgdG8gYmUgcGFja2VkJyk7XG4gICAgLy8gUmVzdG9yZSB0aGUgaW5wdXQgc2hhcGUgdG8gb3JpZ2luYWwuXG4gICAgeFRleERhdGEuc2hhcGUgPSBvcmlnaW5hbFhUZXhEYXRhU2hhcGU7XG4gICAgLy8gU2V0IHRoZSBvdXRwdXQgc2hhcGUgLSB0aGVyZSBpcyBubyBuZWVkIGZvciBleHBlbnNpdmUgcmVzaGFwZSBhcyBkYXRhXG4gICAgLy8gbGF5b3V0IGlzIGFscmVhZHkgY29ycmVjdC5cbiAgICBwb2ludHdpc2VDb252VGV4RGF0YS5zaGFwZSA9IGNvbnZJbmZvLm91dFNoYXBlO1xuXG4gICAgb3V0ID0gaWRlbnRpdHkoe2lucHV0czoge3g6IHBvaW50d2lzZUNvbnZ9LCBiYWNrZW5kfSk7XG4gICAgb3V0LnNoYXBlID0gY29udkluZm8ub3V0U2hhcGU7XG5cbiAgICBpbnRlcm1lZGlhdGVzLnB1c2gocG9pbnR3aXNlQ29udik7XG4gIH0gZWxzZSB7XG4gICAgY29uc3QgbnVtQ29scyA9IGNvbnZJbmZvLm91dEhlaWdodCAqIGNvbnZJbmZvLm91dFdpZHRoO1xuICAgIGNvbnN0IHhSZXNoYXBlZCA9IHJlc2hhcGUoe1xuICAgICAgaW5wdXRzOiB7eH0sXG4gICAgICBiYWNrZW5kLFxuICAgICAgYXR0cnM6IHtcbiAgICAgICAgc2hhcGU6IGlzQ2hhbm5lbHNMYXN0ID9cbiAgICAgICAgICAgIFtjb252SW5mby5iYXRjaFNpemUsIG51bUNvbHMsIGNvbnZJbmZvLmluQ2hhbm5lbHNdIDpcbiAgICAgICAgICAgIFtjb252SW5mby5iYXRjaFNpemUsIGNvbnZJbmZvLmluQ2hhbm5lbHMsIG51bUNvbHNdXG4gICAgICB9XG4gICAgfSk7XG4gICAgY29uc3QgZmlsdGVyUmVzaGFwZWQgPSByZXNoYXBlKHtcbiAgICAgIGlucHV0czoge3g6IGZpbHRlcn0sXG4gICAgICBiYWNrZW5kLFxuICAgICAgYXR0cnM6IHtzaGFwZTogWzEsIGNvbnZJbmZvLmluQ2hhbm5lbHMsIGNvbnZJbmZvLm91dENoYW5uZWxzXX1cbiAgICB9KTtcbiAgICBjb25zdCByZXN1bHQgPSBiYXRjaE1hdE11bEltcGwoe1xuICAgICAgYTogaXNDaGFubmVsc0xhc3QgPyB4UmVzaGFwZWQgOiBmaWx0ZXJSZXNoYXBlZCxcbiAgICAgIGI6IGlzQ2hhbm5lbHNMYXN0ID8gZmlsdGVyUmVzaGFwZWQgOiB4UmVzaGFwZWQsXG4gICAgICB0cmFuc3Bvc2VBOiAhaXNDaGFubmVsc0xhc3QsXG4gICAgICB0cmFuc3Bvc2VCLFxuICAgICAgYmFja2VuZCxcbiAgICAgIGJpYXMsXG4gICAgICBhY3RpdmF0aW9uLFxuICAgICAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyxcbiAgICAgIGxlYWt5cmVsdUFscGhhXG4gICAgfSk7XG5cbiAgICBvdXQgPSByZXNoYXBlKFxuICAgICAgICB7aW5wdXRzOiB7eDogcmVzdWx0fSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogY29udkluZm8ub3V0U2hhcGV9fSk7XG5cbiAgICBpbnRlcm1lZGlhdGVzLnB1c2goeFJlc2hhcGVkKTtcbiAgICBpbnRlcm1lZGlhdGVzLnB1c2goZmlsdGVyUmVzaGFwZWQpO1xuICAgIGludGVybWVkaWF0ZXMucHVzaChyZXN1bHQpO1xuICB9XG5cbiAgZm9yIChjb25zdCBpIG9mIGludGVybWVkaWF0ZXMpIHtcbiAgICBiYWNrZW5kLmRpc3Bvc2VJbnRlcm1lZGlhdGVUZW5zb3JJbmZvKGkpO1xuICB9XG5cbiAgcmV0dXJuIG91dDtcbn1cblxuLy8gSW1wbGVtZW50cyB0aGUgaW0ycm93IGFsZ29yaXRobSBhcyBvdXRsaW5lZCBpbiBcIkhpZ2ggUGVyZm9ybWFuY2Vcbi8vIENvbnZvbHV0aW9uYWwgTmV1cmFsIE5ldHdvcmtzIGZvciBEb2N1bWVudCBQcm9jZXNzaW5nXCIgKFN1dmlzb2Z0LCAyMDA2KVxuZXhwb3J0IGZ1bmN0aW9uIGNvbnYyZFdpdGhJbTJSb3coe1xuICB4LFxuICBmaWx0ZXIsXG4gIGNvbnZJbmZvLFxuICBiYWNrZW5kLFxuICBiaWFzID0gbnVsbCxcbiAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyA9IG51bGwsXG4gIGxlYWt5cmVsdUFscGhhID0gMCxcbiAgYWN0aXZhdGlvbiA9IG51bGxcbn06IENvbnYyRENvbmZpZykge1xuICAvLyBSZWFycmFuZ2VzIGNvbnYyZCBpbnB1dCBzbyBlYWNoIGJsb2NrIHRvIGJlIGNvbnZvbHZlZCBvdmVyIGZvcm1zIHRoZVxuICAvLyBjb2x1bW4gb2YgYSBuZXcgbWF0cml4IHdpdGggc2hhcGUgW2ZpbHRlcldpZHRoICogZmlsdGVySGVpZ2h0ICpcbiAgLy8gaW5DaGFubmVscywgb3V0SGVpZ2h0ICogb3V0V2lkdGhdLiBUaGUgZmlsdGVyIGlzIGFsc28gcmVhcnJhbmdlZCBzbyBlYWNoXG4gIC8vIG91dHB1dCBjaGFubmVsIGZvcm1zIGEgcm93IG9mIGEgbmV3IG1hdHJpeCB3aXRoIHNoYXBlIFtvdXRDaGFubmVscyxcbiAgLy8gZmlsdGVyV2lkdGggKiBmaWx0ZXJIZWlnaHQgKiBpbkNoYW5uZWxzXS4gVGhlIGNvbnZvbHV0aW9uIGlzIHRoZW5cbiAgLy8gY29tcHV0ZWQgYnkgbXVsdGlwbHlpbmcgdGhlc2UgbWF0cmljZXMgYW5kIHJlc2hhcGluZyB0aGUgcmVzdWx0LlxuICBjb25zdCB7XG4gICAgZmlsdGVyV2lkdGgsXG4gICAgZmlsdGVySGVpZ2h0LFxuICAgIGluQ2hhbm5lbHMsXG4gICAgb3V0V2lkdGgsXG4gICAgb3V0SGVpZ2h0LFxuICAgIGRhdGFGb3JtYXRcbiAgfSA9IGNvbnZJbmZvO1xuXG4gIGNvbnN0IGlzQ2hhbm5lbHNMYXN0ID0gZGF0YUZvcm1hdCA9PT0gJ2NoYW5uZWxzTGFzdCc7XG5cbiAgY29uc3Qgc2hhcmVkRGltID0gZmlsdGVyV2lkdGggKiBmaWx0ZXJIZWlnaHQgKiBpbkNoYW5uZWxzO1xuICBjb25zdCBudW1Db2xzID0gb3V0SGVpZ2h0ICogb3V0V2lkdGg7XG4gIGNvbnN0IHgyQ29sU2hhcGUgPSBbY29udkluZm8uYmF0Y2hTaXplLCBzaGFyZWREaW0sIG51bUNvbHNdO1xuICBjb25zdCB0cmFuc3Bvc2VBID0gdHJ1ZTtcbiAgY29uc3QgdHJhbnNwb3NlQiA9IGZhbHNlO1xuXG4gIGNvbnN0IGludGVybWVkaWF0ZXM6IFRlbnNvckluZm9bXSA9IFtdO1xuXG4gIGlmIChwcmVsdUFjdGl2YXRpb25XZWlnaHRzICE9IG51bGwpIHtcbiAgICBjb25zdCB0YXJnZXRTaGFwZSA9XG4gICAgICAgIGdldFNoYXBlRm9yQmF0Y2hNYXRNdWwocHJlbHVBY3RpdmF0aW9uV2VpZ2h0cy5zaGFwZSwgaXNDaGFubmVsc0xhc3QpO1xuICAgIGlmICh0YXJnZXRTaGFwZSAhPSBudWxsKSB7XG4gICAgICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gcmVzaGFwZSh7XG4gICAgICAgIGlucHV0czoge3g6IHByZWx1QWN0aXZhdGlvbldlaWdodHN9LFxuICAgICAgICBiYWNrZW5kLFxuICAgICAgICBhdHRyczoge3NoYXBlOiB0YXJnZXRTaGFwZX1cbiAgICAgIH0pO1xuICAgICAgaW50ZXJtZWRpYXRlcy5wdXNoKHByZWx1QWN0aXZhdGlvbldlaWdodHMpO1xuICAgIH1cbiAgfVxuXG4gIGlmIChiaWFzICE9IG51bGwpIHtcbiAgICBjb25zdCB0YXJnZXRTaGFwZSA9IGdldFNoYXBlRm9yQmF0Y2hNYXRNdWwoYmlhcy5zaGFwZSwgaXNDaGFubmVsc0xhc3QpO1xuICAgIGlmICh0YXJnZXRTaGFwZSAhPSBudWxsKSB7XG4gICAgICBiaWFzID0gcmVzaGFwZSh7aW5wdXRzOiB7eDogYmlhc30sIGJhY2tlbmQsIGF0dHJzOiB7c2hhcGU6IHRhcmdldFNoYXBlfX0pO1xuICAgICAgaW50ZXJtZWRpYXRlcy5wdXNoKGJpYXMpO1xuICAgIH1cbiAgfVxuXG4gIGNvbnN0IHcyUm93ID0gcmVzaGFwZSh7XG4gICAgaW5wdXRzOiB7eDogZmlsdGVyfSxcbiAgICBiYWNrZW5kLFxuICAgIGF0dHJzOiB7c2hhcGU6IFsxLCBzaGFyZWREaW0sIHV0aWwuc2l6ZUZyb21TaGFwZShmaWx0ZXIuc2hhcGUpIC8gc2hhcmVkRGltXX1cbiAgfSk7XG4gIGludGVybWVkaWF0ZXMucHVzaCh3MlJvdyk7XG5cbiAgY29uc3QgaW0yQ29sUHJvZ3JhbSA9IG5ldyBJbTJDb2xQYWNrZWRQcm9ncmFtKHgyQ29sU2hhcGUsIGNvbnZJbmZvKTtcbiAgY29uc3QgY3VzdG9tVmFsdWVzID0gW1xuICAgIHguc2hhcGUsIFtjb252SW5mby5wYWRJbmZvLnRvcCwgY29udkluZm8ucGFkSW5mby5sZWZ0XSxcbiAgICBbY29udkluZm8uc3RyaWRlSGVpZ2h0LCBjb252SW5mby5zdHJpZGVXaWR0aF0sXG4gICAgW2NvbnZJbmZvLmRpbGF0aW9uSGVpZ2h0LCBjb252SW5mby5kaWxhdGlvbldpZHRoXSwgW2NvbnZJbmZvLmluQ2hhbm5lbHNdLFxuICAgIFtjb252SW5mby5maWx0ZXJXaWR0aCAqIGNvbnZJbmZvLmluQ2hhbm5lbHNdLCBbY29udkluZm8ub3V0V2lkdGhdXG4gIF07XG4gIGNvbnN0IGltMkNvbCA9XG4gICAgICBiYWNrZW5kLnJ1bldlYkdMUHJvZ3JhbShpbTJDb2xQcm9ncmFtLCBbeF0sICdmbG9hdDMyJywgY3VzdG9tVmFsdWVzKTtcbiAgY29uc3QgaW0yQ29sUmVzaGFwZWQgPVxuICAgICAgcmVzaGFwZSh7aW5wdXRzOiB7eDogaW0yQ29sfSwgYmFja2VuZCwgYXR0cnM6IHtzaGFwZTogeDJDb2xTaGFwZX19KTtcblxuICBpbnRlcm1lZGlhdGVzLnB1c2goaW0yQ29sKTtcbiAgaW50ZXJtZWRpYXRlcy5wdXNoKGltMkNvbFJlc2hhcGVkKTtcblxuICBjb25zdCBoYXNCaWFzID0gYmlhcyAhPSBudWxsO1xuICBjb25zdCBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyAhPSBudWxsO1xuICBjb25zdCBoYXNMZWFreXJlbHVBbHBoYSA9IGFjdGl2YXRpb24gPT09ICdsZWFreXJlbHUnO1xuICBjb25zdCBmdXNlZEFjdGl2YXRpb24gPVxuICAgICAgYWN0aXZhdGlvbiA/IG1hcEFjdGl2YXRpb25Ub1NoYWRlclByb2dyYW0oYWN0aXZhdGlvbiwgdHJ1ZSkgOiBudWxsO1xuICBjb25zdCBtYXRtdWxQcm9ncmFtID0gbmV3IE1hdE11bFBhY2tlZFByb2dyYW0oXG4gICAgICBpc0NoYW5uZWxzTGFzdCA/IGltMkNvbFJlc2hhcGVkLnNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA6XG4gICAgICAgICAgICAgICAgICAgICAgIHcyUm93LnNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgIGlzQ2hhbm5lbHNMYXN0ID8gdzJSb3cuc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdIDpcbiAgICAgICAgICAgICAgICAgICAgICAgaW0yQ29sUmVzaGFwZWQuc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgICAgaXNDaGFubmVsc0xhc3QgPyBbY29udkluZm8uYmF0Y2hTaXplLCBudW1Db2xzLCBjb252SW5mby5vdXRDaGFubmVsc10gOlxuICAgICAgICAgICAgICAgICAgICAgICBbY29udkluZm8uYmF0Y2hTaXplLCBjb252SW5mby5vdXRDaGFubmVscywgbnVtQ29sc10sXG4gICAgICB0cmFuc3Bvc2VBLCB0cmFuc3Bvc2VCLCBoYXNCaWFzLCBmdXNlZEFjdGl2YXRpb24sXG4gICAgICBoYXNQcmVsdUFjdGl2YXRpb25XZWlnaHRzLCBoYXNMZWFreXJlbHVBbHBoYSk7XG4gIGNvbnN0IGlucHV0czogVGVuc29ySW5mb1tdID1cbiAgICAgIGlzQ2hhbm5lbHNMYXN0ID8gW2ltMkNvbFJlc2hhcGVkLCB3MlJvd10gOiBbdzJSb3csIGltMkNvbFJlc2hhcGVkXTtcbiAgaWYgKGJpYXMpIHtcbiAgICBpbnB1dHMucHVzaChiaWFzKTtcbiAgfVxuICBpZiAoaGFzUHJlbHVBY3RpdmF0aW9uV2VpZ2h0cykge1xuICAgIGlucHV0cy5wdXNoKHByZWx1QWN0aXZhdGlvbldlaWdodHMpO1xuICB9XG4gIGlmIChoYXNMZWFreXJlbHVBbHBoYSkge1xuICAgIGNvbnN0ICRsZWFreXJlbHVBbHBoYSA9IGJhY2tlbmQubWFrZVRlbnNvckluZm8oXG4gICAgICAgIFtdLCAnZmxvYXQzMicsXG4gICAgICAgIHV0aWwuY3JlYXRlU2NhbGFyVmFsdWUobGVha3lyZWx1QWxwaGEgYXMgdW5rbm93biBhcyAnZmxvYXQzMicsXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ2Zsb2F0MzInKSk7XG4gICAgaW5wdXRzLnB1c2goJGxlYWt5cmVsdUFscGhhKTtcbiAgICBpbnRlcm1lZGlhdGVzLnB1c2goJGxlYWt5cmVsdUFscGhhKTtcbiAgfVxuICBjb25zdCBwcm9kdWN0ID0gYmFja2VuZC5ydW5XZWJHTFByb2dyYW0obWF0bXVsUHJvZ3JhbSwgaW5wdXRzLCAnZmxvYXQzMicpO1xuICBjb25zdCBvdXQgPSByZXNoYXBlKFxuICAgICAge2lucHV0czoge3g6IHByb2R1Y3R9LCBiYWNrZW5kLCBhdHRyczoge3NoYXBlOiBjb252SW5mby5vdXRTaGFwZX19KTtcblxuICBpbnRlcm1lZGlhdGVzLnB1c2gocHJvZHVjdCk7XG4gIGZvciAoY29uc3QgaSBvZiBpbnRlcm1lZGlhdGVzKSB7XG4gICAgYmFja2VuZC5kaXNwb3NlSW50ZXJtZWRpYXRlVGVuc29ySW5mbyhpKTtcbiAgfVxuXG4gIHJldHVybiBvdXQ7XG59XG4iXX0=