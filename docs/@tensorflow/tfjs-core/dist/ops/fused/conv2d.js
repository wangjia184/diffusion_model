/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../../engine';
import { customGrad } from '../../gradients';
import { FusedConv2D } from '../../kernel_names';
import { makeTypesMatch } from '../../tensor_util';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { add } from '../add';
import * as broadcast_util from '../broadcast_util';
import { conv2d as unfusedConv2d } from '../conv2d';
import { conv2DBackpropFilter } from '../conv2d_backprop_filter';
import { conv2DBackpropInput } from '../conv2d_backprop_input';
import * as conv_util from '../conv_util';
import { applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse } from '../fused_util';
import { op } from '../operation';
import { reshape } from '../reshape';
/**
 * Computes a 2D convolution over the input x, optionally fused with adding a
 * bias and applying an activation.
 *
 * ```js
 * const inputDepth = 2;
 * const inShape = [2, 2, 2, inputDepth];
 * const outputDepth = 2;
 * const fSize = 1;
 * const pad = 0;
 * const strides = 1;
 *
 * const x = tf.tensor4d( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
 * 16], inShape);
 * const w = tf.tensor4d([-1, 1, -2, 0.5], [fSize, fSize, inputDepth,
 * outputDepth]);
 *
 * tf.fused.conv2d({ x, filter: w, strides, pad, dataFormat: 'NHWC',
 * dilations: [1, 1], bias: tf.scalar(5), activation: 'relu' }).print();
 * ```
 *
 * @param obj An object with the following properties:
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid` output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dataFormat An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `dilations` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`) to be
 *     applied
 *      after biasAdd.
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 * @param leakyreluAlpha Optional. Alpha to be applied as part of a `leakyrelu`
 *     activation.
 */
function fusedConv2d_({ x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode, bias, activation = 'linear', preluActivationWeights, leakyreluAlpha }) {
    activation = activation || 'linear';
    if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
        // TODO: Transpose bias and preluActivationWeights properly for NCHW
        // format before computation.
        util.assert(dataFormat === 'NHWC', () => `Error in fused conv2d: got dataFormat of ${dataFormat} but ` +
            `only NHWC is currently supported for the case of gradient depth ` +
            `is 0 and the activation is not linear.`);
        let result = unfusedConv2d(x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
        if (bias != null) {
            result = add(result, bias);
        }
        return applyActivation(result, activation, preluActivationWeights, leakyreluAlpha);
    }
    const $x = convertToTensor(x, 'x', 'conv2d', 'float32');
    const $filter = convertToTensor(filter, 'filter', 'conv2d', 'float32');
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    util.assert(x4D.rank === 4, () => `Error in fused conv2d: input must be rank 4, but got rank ` +
        `${x4D.rank}.`);
    util.assert($filter.rank === 4, () => `Error in fused conv2d: filter must be rank 4, but got rank ` +
        `${$filter.rank}.`);
    conv_util.checkPadOnDimRoundingMode('fused conv2d', pad, dimRoundingMode);
    const inputChannels = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
    util.assert($filter.shape[2] === inputChannels, () => `Error in conv2d: depth of input (${inputChannels}) must match ` +
        `input depth for filter ${$filter.shape[2]}.`);
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in conv2D: Either strides or dilations must be 1. ' +
        `Got strides ${strides} and dilations '${dilations}'`);
    const convInfo = conv_util.computeConv2DInfo(x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode);
    let $bias;
    if (bias != null) {
        $bias = convertToTensor(bias, 'bias', 'fused conv2d');
        [$bias] = makeTypesMatch($bias, $x);
        // According to TensorFlow, the bias is supposed be a 1-D tensor or a
        // scalar.
        //
        // 3-D or 4-D bias is not disabled for NHWC format, because they are
        // currently being used in some cases. For examplem in our code base,
        // https://github.com/tensorflow/tfjs/blob/b53bd47e880367ae57493f0ea628abaf08db2d5d/tfjs-core/src/ops/fused/fused_conv2d_test.ts#L1972.
        if (dataFormat === 'NHWC') {
            broadcast_util.assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
        }
        else {
            util.assert($bias.shape.length <= 1, () => `Error in fused conv2d: only supports scalar or 1-D Tensor ` +
                `bias for NCHW format but got the bias of ` +
                `rank-${$bias.shape.length}.`);
            util.assert($bias.shape.length === 0 || $bias.shape[0] === convInfo.outChannels ||
                $bias.shape[0] === 1, () => `Error in fused conv2d: bias shape (${$bias.shape}) is not ` +
                `compatible with the number of output channels ` +
                `(${convInfo.outChannels})`);
        }
    }
    let $preluActivationWeights;
    if (preluActivationWeights != null) {
        // PReLU's activation weights could be a scalar, a 1-D tensor or a 3-D
        // tensor.
        const alphaShape = preluActivationWeights.shape;
        util.assert(alphaShape.length <= 1 || alphaShape.length === 3, () => `Error in fused conv2d: only supports scalar, 1-D Tensor or ` +
            `3-D Tensor PReLU activation weights but got a tensor of ` +
            `rank-${alphaShape.length}.`);
        if (alphaShape.length === 1) {
            // Whether the data format is NCHW or NHWC, the 1-D PReLU activation
            // weights tensor should be aligned with the output channels of conv2d
            // result.
            util.assert(alphaShape[0] === 1 || alphaShape[0] === convInfo.outChannels, () => `Error in fused conv2d: PReLU activation weights ` +
                `(${alphaShape}) is not compatible with the number of output ` +
                `channels (${convInfo.outChannels}).`);
        }
        else if (alphaShape.length === 3) {
            // Whether the data format is NCHW or NHWC, the PReLU activation weights
            // tensor should has the compatible shape with the result of conv2d.
            try {
                broadcast_util.assertAndGetBroadcastShape(alphaShape, convInfo.outShape);
            }
            catch (e) {
                const errMsg = `Error in fused conv2d: PReLU activation weights (${alphaShape}) ` +
                    `is not compatible with the output shape of the conv2d ` +
                    `(${convInfo.outShape}).`;
                throw Error(errMsg);
            }
        }
        $preluActivationWeights = convertToTensor(preluActivationWeights, 'prelu weights', 'fused conv2d');
    }
    const grad = (dy, saved) => {
        util.assert(dataFormat === 'NHWC', () => `Error in gradient of fused conv2D: got dataFormat of ${dataFormat} but only NHWC is currently supported.`);
        const [$filter, x4D, y, $bias] = saved;
        const dyActivation = getFusedDyActivation(dy, y, activation);
        util.assert(conv_util.tupleValuesAreOne(dilations), () => 'Error in gradient of fused conv2D: ' +
            `dilation rates greater than 1 ` +
            `are not yet supported in gradients. Got dilations '${dilations}'`);
        const xDer = conv2DBackpropInput(x4D.shape, dyActivation, $filter, strides, pad);
        const filterDer = conv2DBackpropFilter(x4D, dyActivation, $filter.shape, strides, pad);
        const der = [xDer, filterDer];
        if ($bias != null) {
            const biasDer = getFusedBiasGradient($bias, dyActivation);
            der.push(biasDer);
        }
        return der;
    };
    const inputs = {
        x: x4D,
        filter: $filter,
        bias: $bias,
        preluActivationWeights: $preluActivationWeights
    };
    const attrs = {
        strides,
        pad,
        dataFormat,
        dilations,
        dimRoundingMode,
        activation,
        leakyreluAlpha
    };
    // Depending on the the params passed in we will have different number of
    // inputs and thus a a different number of elements in the gradient.
    if (bias == null) {
        const customOp = customGrad((x4D, filter, save) => {
            let res = 
            // tslint:disable-next-line: no-unnecessary-type-assertion
            ENGINE.runKernel(FusedConv2D, inputs, attrs);
            save([filter, x4D, res]);
            if (reshapedTo4D) {
                // tslint:disable-next-line: no-unnecessary-type-assertion
                res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
            }
            return { value: res, gradFunc: grad };
        });
        return customOp(x4D, $filter);
    }
    else {
        const customOpWithBias = customGrad((x4D, filter, bias, save) => {
            let res = ENGINE.runKernel(FusedConv2D, inputs, attrs);
            save([filter, x4D, res, bias]);
            if (reshapedTo4D) {
                // tslint:disable-next-line: no-unnecessary-type-assertion
                res = reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
            }
            return { value: res, gradFunc: grad };
        });
        return customOpWithBias(x4D, $filter, $bias);
    }
}
export const conv2d = /* @__PURE__ */ op({ fusedConv2d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjJkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZnVzZWQvY29udjJkLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLFVBQVUsRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBQzNDLE9BQU8sRUFBQyxXQUFXLEVBQXNDLE1BQU0sb0JBQW9CLENBQUM7QUFJcEYsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLG1CQUFtQixDQUFDO0FBQ2pELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSx1QkFBdUIsQ0FBQztBQUV0RCxPQUFPLEtBQUssSUFBSSxNQUFNLFlBQVksQ0FBQztBQUNuQyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzNCLE9BQU8sS0FBSyxjQUFjLE1BQU0sbUJBQW1CLENBQUM7QUFDcEQsT0FBTyxFQUFDLE1BQU0sSUFBSSxhQUFhLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEQsT0FBTyxFQUFDLG9CQUFvQixFQUFDLE1BQU0sMkJBQTJCLENBQUM7QUFDL0QsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0sMEJBQTBCLENBQUM7QUFDN0QsT0FBTyxLQUFLLFNBQVMsTUFBTSxjQUFjLENBQUM7QUFFMUMsT0FBTyxFQUFDLGVBQWUsRUFBRSxvQkFBb0IsRUFBRSxvQkFBb0IsRUFBRSxVQUFVLEVBQUMsTUFBTSxlQUFlLENBQUM7QUFDdEcsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRW5DOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXdERztBQUNILFNB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LENBQUM7S0FDcEM7U0FBTTtRQUNMLE1BQU0sZ0JBQWdCLEdBQUcsVUFBVSxDQUMvQixDQUFDLEdBQWEsRUFBRSxNQUFnQixFQUFFLElBQVksRUFBRSxJQUFrQixFQUFFLEVBQUU7WUFDcEUsSUFBSSxHQUFHLEdBQXNCLE1BQU0sQ0FBQyxTQUFTLENBQ3pDLFdBQVcsRUFBRSxNQUFtQyxFQUNoRCxLQUFnQyxDQUFDLENBQUM7WUFFdEMsSUFBSSxDQUFDLENBQUMsTUFBTSxFQUFFLEdBQUcsRUFBRSxHQUFHLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztZQUUvQixJQUFJLFlBQVksRUFBRTtnQkFDaEIsMERBQTBEO2dCQUMxRCxHQUFHLEdBQUcsT0FBTyxDQUFDLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQ2pELENBQUM7YUFDZDtZQUVELE9BQU8sRUFBQyxLQUFLLEVBQUUsR0FBRyxFQUFFLFFBQVEsRUFBRSxJQUFJLEVBQUMsQ0FBQztRQUN0QyxDQUFDLENBQUMsQ0FBQztRQUVQLE9BQU8sZ0JBQWdCLENBQUMsR0FBRyxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQU0sQ0FBQztLQUNuRDtBQUNILENBQUM7QUFDRCxNQUFNLENBQUMsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFlBQVksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtjdXN0b21HcmFkfSBmcm9tICcuLi8uLi9ncmFkaWVudHMnO1xuaW1wb3J0IHtGdXNlZENvbnYyRCwgRnVzZWRDb252MkRBdHRycywgRnVzZWRDb252MkRJbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4vLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yLCBUZW5zb3IzRCwgVGVuc29yNER9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge0dyYWRTYXZlRnVuYywgTmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uLy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge21ha2VUeXBlc01hdGNofSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbCc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi8uLi91dGlsJztcbmltcG9ydCB7YWRkfSBmcm9tICcuLi9hZGQnO1xuaW1wb3J0ICogYXMgYnJvYWRjYXN0X3V0aWwgZnJvbSAnLi4vYnJvYWRjYXN0X3V0aWwnO1xuaW1wb3J0IHtjb252MmQgYXMgdW5mdXNlZENvbnYyZH0gZnJvbSAnLi4vY29udjJkJztcbmltcG9ydCB7Y29udjJEQmFja3Byb3BGaWx0ZXJ9IGZyb20gJy4uL2NvbnYyZF9iYWNrcHJvcF9maWx0ZXInO1xuaW1wb3J0IHtjb252MkRCYWNrcHJvcElucHV0fSBmcm9tICcuLi9jb252MmRfYmFja3Byb3BfaW5wdXQnO1xuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uL2NvbnZfdXRpbCc7XG5pbXBvcnQge0FjdGl2YXRpb259IGZyb20gJy4uL2Z1c2VkX3R5cGVzJztcbmltcG9ydCB7YXBwbHlBY3RpdmF0aW9uLCBnZXRGdXNlZEJpYXNHcmFkaWVudCwgZ2V0RnVzZWREeUFjdGl2YXRpb24sIHNob3VsZEZ1c2V9IGZyb20gJy4uL2Z1c2VkX3V0aWwnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi4vcmVzaGFwZSc7XG5cbi8qKlxuICogQ29tcHV0ZXMgYSAyRCBjb252b2x1dGlvbiBvdmVyIHRoZSBpbnB1dCB4LCBvcHRpb25hbGx5IGZ1c2VkIHdpdGggYWRkaW5nIGFcbiAqIGJpYXMgYW5kIGFwcGx5aW5nIGFuIGFjdGl2YXRpb24uXG4gKlxuICogYGBganNcbiAqIGNvbnN0IGlucHV0RGVwdGggPSAyO1xuICogY29uc3QgaW5TaGFwZSA9IFsyLCAyLCAyLCBpbnB1dERlcHRoXTtcbiAqIGNvbnN0IG91dHB1dERlcHRoID0gMjtcbiAqIGNvbnN0IGZTaXplID0gMTtcbiAqIGNvbnN0IHBhZCA9IDA7XG4gKiBjb25zdCBzdHJpZGVzID0gMTtcbiAqXG4gKiBjb25zdCB4ID0gdGYudGVuc29yNGQoIFsxLCAyLCAzLCA0LCA1LCA2LCA3LCA4LCA5LCAxMCwgMTEsIDEyLCAxMywgMTQsIDE1LFxuICogMTZdLCBpblNoYXBlKTtcbiAqIGNvbnN0IHcgPSB0Zi50ZW5zb3I0ZChbLTEsIDEsIC0yLCAwLjVdLCBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLFxuICogb3V0cHV0RGVwdGhdKTtcbiAqXG4gKiB0Zi5mdXNlZC5jb252MmQoeyB4LCBmaWx0ZXI6IHcsIHN0cmlkZXMsIHBhZCwgZGF0YUZvcm1hdDogJ05IV0MnLFxuICogZGlsYXRpb25zOiBbMSwgMV0sIGJpYXM6IHRmLnNjYWxhcig1KSwgYWN0aXZhdGlvbjogJ3JlbHUnIH0pLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gb2JqIEFuIG9iamVjdCB3aXRoIHRoZSBmb2xsb3dpbmcgcHJvcGVydGllczpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCB0ZW5zb3IsIG9mIHJhbmsgNCBvciByYW5rIDMsIG9mIHNoYXBlXG4gKiAgICAgYFtiYXRjaCwgaGVpZ2h0LCB3aWR0aCwgaW5DaGFubmVsc11gLiBJZiByYW5rIDMsIGJhdGNoIG9mIDEgaXNcbiAqIGFzc3VtZWQuXG4gKiBAcGFyYW0gZmlsdGVyIFRoZSBmaWx0ZXIsIHJhbmsgNCwgb2Ygc2hhcGVcbiAqICAgICBgW2ZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGgsIGluRGVwdGgsIG91dERlcHRoXWAuXG4gKiBAcGFyYW0gc3RyaWRlcyBUaGUgc3RyaWRlcyBvZiB0aGUgY29udm9sdXRpb246IGBbc3RyaWRlSGVpZ2h0LFxuICogc3RyaWRlV2lkdGhdYC5cbiAqIEBwYXJhbSBwYWQgVGhlIHR5cGUgb2YgcGFkZGluZyBhbGdvcml0aG0uXG4gKiAgIC0gYHNhbWVgIGFuZCBzdHJpZGUgMTogb3V0cHV0IHdpbGwgYmUgb2Ygc2FtZSBzaXplIGFzIGlucHV0LFxuICogICAgICAgcmVnYXJkbGVzcyBvZiBmaWx0ZXIgc2l6ZS5cbiAqICAgLSBgdmFsaWRgIG91dHB1dCB3aWxsIGJlIHNtYWxsZXIgdGhhbiBpbnB1dCBpZiBmaWx0ZXIgaXMgbGFyZ2VyXG4gKiAgICAgICB0aGFuIDF4MS5cbiAqICAgLSBGb3IgbW9yZSBpbmZvLCBzZWUgdGhpcyBndWlkZTpcbiAqICAgICBbaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uXShcbiAqICAgICAgICAgIGh0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9jb252b2x1dGlvbilcbiAqIEBwYXJhbSBkYXRhRm9ybWF0IEFuIG9wdGlvbmFsIHN0cmluZyBmcm9tOiBcIk5IV0NcIiwgXCJOQ0hXXCIuIERlZmF1bHRzIHRvXG4gKiAgICAgXCJOSFdDXCIuIFNwZWNpZnkgdGhlIGRhdGEgZm9ybWF0IG9mIHRoZSBpbnB1dCBhbmQgb3V0cHV0IGRhdGEuIFdpdGggdGhlXG4gKiAgICAgZGVmYXVsdCBmb3JtYXQgXCJOSFdDXCIsIHRoZSBkYXRhIGlzIHN0b3JlZCBpbiB0aGUgb3JkZXIgb2Y6IFtiYXRjaCxcbiAqICAgICBoZWlnaHQsIHdpZHRoLCBjaGFubmVsc10uIE9ubHkgXCJOSFdDXCIgaXMgY3VycmVudGx5IHN1cHBvcnRlZC5cbiAqIEBwYXJhbSBkaWxhdGlvbnMgVGhlIGRpbGF0aW9uIHJhdGVzOiBgW2RpbGF0aW9uSGVpZ2h0LCBkaWxhdGlvbldpZHRoXWBcbiAqICAgICBpbiB3aGljaCB3ZSBzYW1wbGUgaW5wdXQgdmFsdWVzIGFjcm9zcyB0aGUgaGVpZ2h0IGFuZCB3aWR0aCBkaW1lbnNpb25zXG4gKiAgICAgaW4gYXRyb3VzIGNvbnZvbHV0aW9uLiBEZWZhdWx0cyB0byBgWzEsIDFdYC4gSWYgYGRpbGF0aW9uc2AgaXMgYSBzaW5nbGVcbiAqICAgICBudW1iZXIsIHRoZW4gYGRpbGF0aW9uSGVpZ2h0ID09IGRpbGF0aW9uV2lkdGhgLiBJZiBpdCBpcyBncmVhdGVyIHRoYW5cbiAqICAgICAxLCB0aGVuIGFsbCB2YWx1ZXMgb2YgYHN0cmlkZXNgIG11c3QgYmUgMS5cbiAqIEBwYXJhbSBkaW1Sb3VuZGluZ01vZGUgQSBzdHJpbmcgZnJvbTogJ2NlaWwnLCAncm91bmQnLCAnZmxvb3InLiBJZiBub25lIGlzXG4gKiAgICAgcHJvdmlkZWQsIGl0IHdpbGwgZGVmYXVsdCB0byB0cnVuY2F0ZS5cbiAqIEBwYXJhbSBiaWFzIFRlbnNvciB0byBiZSBhZGRlZCB0byB0aGUgcmVzdWx0LlxuICogQHBhcmFtIGFjdGl2YXRpb24gTmFtZSBvZiBhY3RpdmF0aW9uIGtlcm5lbCAoZGVmYXVsdHMgdG8gYGxpbmVhcmApIHRvIGJlXG4gKiAgICAgYXBwbGllZFxuICogICAgICBhZnRlciBiaWFzQWRkLlxuICogQHBhcmFtIHByZWx1QWN0aXZhdGlvbldlaWdodHMgVGVuc29yIG9mIHByZWx1IHdlaWdodHMgdG8gYmUgYXBwbGllZCBhcyBwYXJ0XG4gKiAgICAgb2YgYSBgcHJlbHVgIGFjdGl2YXRpb24sIHR5cGljYWxseSB0aGUgc2FtZSBzaGFwZSBhcyBgeGAuXG4gKiBAcGFyYW0gbGVha3lyZWx1QWxwaGEgT3B0aW9uYWwuIEFscGhhIHRvIGJlIGFwcGxpZWQgYXMgcGFydCBvZiBhIGBsZWFreXJlbHVgXG4gKiAgICAgYWN0aXZhdGlvbi5cbiAqL1xuZnVuY3Rpb24gZnVzZWRDb252MmRfPFQgZXh0ZW5kcyBUZW5zb3IzRHxUZW5zb3I0RD4oe1xuICB4LFxuICBmaWx0ZXIsXG4gIHN0cmlkZXMsXG4gIHBhZCxcbiAgZGF0YUZvcm1hdCA9ICdOSFdDJyxcbiAgZGlsYXRpb25zID0gWzEsIDFdLFxuICBkaW1Sb3VuZGluZ01vZGUsXG4gIGJpYXMsXG4gIGFjdGl2YXRpb24gPSAnbGluZWFyJyxcbiAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyxcbiAgbGVha3lyZWx1QWxwaGFcbn06IHtcbiAgeDogVHxUZW5zb3JMaWtlLFxuICBmaWx0ZXI6IFRlbnNvcjREfFRlbnNvckxpa2UsXG4gIHN0cmlkZXM6IFtudW1iZXIsIG51bWJlcl18bnVtYmVyLFxuICBwYWQ6ICd2YWxpZCd8J3NhbWUnfG51bWJlcnxjb252X3V0aWwuRXhwbGljaXRQYWRkaW5nLFxuICBkYXRhRm9ybWF0PzogJ05IV0MnfCdOQ0hXJyxcbiAgZGlsYXRpb25zPzogW251bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gIGRpbVJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcsXG4gIGJpYXM/OiBUZW5zb3J8VGVuc29yTGlrZSxcbiAgYWN0aXZhdGlvbj86IEFjdGl2YXRpb24sXG4gIHByZWx1QWN0aXZhdGlvbldlaWdodHM/OiBUZW5zb3IsXG4gIGxlYWt5cmVsdUFscGhhPzogbnVtYmVyXG59KTogVCB7XG4gIGFjdGl2YXRpb24gPSBhY3RpdmF0aW9uIHx8ICdsaW5lYXInO1xuXG4gIGlmIChzaG91bGRGdXNlKEVOR0lORS5zdGF0ZS5ncmFkaWVudERlcHRoLCBhY3RpdmF0aW9uKSA9PT0gZmFsc2UpIHtcbiAgICAvLyBUT0RPOiBUcmFuc3Bvc2UgYmlhcyBhbmQgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyBwcm9wZXJseSBmb3IgTkNIV1xuICAgIC8vIGZvcm1hdCBiZWZvcmUgY29tcHV0YXRpb24uXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRhdGFGb3JtYXQgPT09ICdOSFdDJyxcbiAgICAgICAgKCkgPT4gYEVycm9yIGluIGZ1c2VkIGNvbnYyZDogZ290IGRhdGFGb3JtYXQgb2YgJHtkYXRhRm9ybWF0fSBidXQgYCArXG4gICAgICAgICAgICBgb25seSBOSFdDIGlzIGN1cnJlbnRseSBzdXBwb3J0ZWQgZm9yIHRoZSBjYXNlIG9mIGdyYWRpZW50IGRlcHRoIGAgK1xuICAgICAgICAgICAgYGlzIDAgYW5kIHRoZSBhY3RpdmF0aW9uIGlzIG5vdCBsaW5lYXIuYCk7XG5cbiAgICBsZXQgcmVzdWx0ID0gdW5mdXNlZENvbnYyZChcbiAgICAgICAgeCwgZmlsdGVyLCBzdHJpZGVzLCBwYWQsIGRhdGFGb3JtYXQsIGRpbGF0aW9ucywgZGltUm91bmRpbmdNb2RlKTtcbiAgICBpZiAoYmlhcyAhPSBudWxsKSB7XG4gICAgICByZXN1bHQgPSBhZGQocmVzdWx0LCBiaWFzKTtcbiAgICB9XG5cbiAgICByZXR1cm4gYXBwbHlBY3RpdmF0aW9uKFxuICAgICAgICAgICAgICAgcmVzdWx0LCBhY3RpdmF0aW9uLCBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLCBsZWFreXJlbHVBbHBoYSkgYXMgVDtcbiAgfVxuXG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2NvbnYyZCcsICdmbG9hdDMyJyk7XG4gIGNvbnN0ICRmaWx0ZXIgPSBjb252ZXJ0VG9UZW5zb3IoZmlsdGVyLCAnZmlsdGVyJywgJ2NvbnYyZCcsICdmbG9hdDMyJyk7XG5cbiAgbGV0IHg0RCA9ICR4IGFzIFRlbnNvcjREO1xuICBsZXQgcmVzaGFwZWRUbzREID0gZmFsc2U7XG5cbiAgaWYgKCR4LnJhbmsgPT09IDMpIHtcbiAgICByZXNoYXBlZFRvNEQgPSB0cnVlO1xuICAgIHg0RCA9IHJlc2hhcGUoJHgsIFsxLCAkeC5zaGFwZVswXSwgJHguc2hhcGVbMV0sICR4LnNoYXBlWzJdXSk7XG4gIH1cbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4NEQucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBmdXNlZCBjb252MmQ6IGlucHV0IG11c3QgYmUgcmFuayA0LCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgYCR7eDRELnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICRmaWx0ZXIucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBmdXNlZCBjb252MmQ6IGZpbHRlciBtdXN0IGJlIHJhbmsgNCwgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgIGAkeyRmaWx0ZXIucmFua30uYCk7XG4gIGNvbnZfdXRpbC5jaGVja1BhZE9uRGltUm91bmRpbmdNb2RlKCdmdXNlZCBjb252MmQnLCBwYWQsIGRpbVJvdW5kaW5nTW9kZSk7XG4gIGNvbnN0IGlucHV0Q2hhbm5lbHMgPSBkYXRhRm9ybWF0ID09PSAnTkhXQycgPyB4NEQuc2hhcGVbM10gOiB4NEQuc2hhcGVbMV07XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGZpbHRlci5zaGFwZVsyXSA9PT0gaW5wdXRDaGFubmVscyxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjb252MmQ6IGRlcHRoIG9mIGlucHV0ICgke2lucHV0Q2hhbm5lbHN9KSBtdXN0IG1hdGNoIGAgK1xuICAgICAgICAgIGBpbnB1dCBkZXB0aCBmb3IgZmlsdGVyICR7JGZpbHRlci5zaGFwZVsyXX0uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgY29udl91dGlsLmVpdGhlclN0cmlkZXNPckRpbGF0aW9uc0FyZU9uZShzdHJpZGVzLCBkaWxhdGlvbnMpLFxuICAgICAgKCkgPT4gJ0Vycm9yIGluIGNvbnYyRDogRWl0aGVyIHN0cmlkZXMgb3IgZGlsYXRpb25zIG11c3QgYmUgMS4gJyArXG4gICAgICAgICAgYEdvdCBzdHJpZGVzICR7c3RyaWRlc30gYW5kIGRpbGF0aW9ucyAnJHtkaWxhdGlvbnN9J2ApO1xuXG4gIGNvbnN0IGNvbnZJbmZvID0gY29udl91dGlsLmNvbXB1dGVDb252MkRJbmZvKFxuICAgICAgeDRELnNoYXBlLCAkZmlsdGVyLnNoYXBlLCBzdHJpZGVzLCBkaWxhdGlvbnMsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcblxuICBsZXQgJGJpYXM6IFRlbnNvcjtcbiAgaWYgKGJpYXMgIT0gbnVsbCkge1xuICAgICRiaWFzID0gY29udmVydFRvVGVuc29yKGJpYXMsICdiaWFzJywgJ2Z1c2VkIGNvbnYyZCcpO1xuICAgIFskYmlhc10gPSBtYWtlVHlwZXNNYXRjaCgkYmlhcywgJHgpO1xuXG4gICAgLy8gQWNjb3JkaW5nIHRvIFRlbnNvckZsb3csIHRoZSBiaWFzIGlzIHN1cHBvc2VkIGJlIGEgMS1EIHRlbnNvciBvciBhXG4gICAgLy8gc2NhbGFyLlxuICAgIC8vXG4gICAgLy8gMy1EIG9yIDQtRCBiaWFzIGlzIG5vdCBkaXNhYmxlZCBmb3IgTkhXQyBmb3JtYXQsIGJlY2F1c2UgdGhleSBhcmVcbiAgICAvLyBjdXJyZW50bHkgYmVpbmcgdXNlZCBpbiBzb21lIGNhc2VzLiBGb3IgZXhhbXBsZW0gaW4gb3VyIGNvZGUgYmFzZSxcbiAgICAvLyBodHRwczovL2dpdGh1Yi5jb20vdGVuc29yZmxvdy90ZmpzL2Jsb2IvYjUzYmQ0N2U4ODAzNjdhZTU3NDkzZjBlYTYyOGFiYWYwOGRiMmQ1ZC90ZmpzLWNvcmUvc3JjL29wcy9mdXNlZC9mdXNlZF9jb252MmRfdGVzdC50cyNMMTk3Mi5cbiAgICBpZiAoZGF0YUZvcm1hdCA9PT0gJ05IV0MnKSB7XG4gICAgICBicm9hZGNhc3RfdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RTaGFwZShjb252SW5mby5vdXRTaGFwZSwgJGJpYXMuc2hhcGUpO1xuICAgIH0gZWxzZSB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICAkYmlhcy5zaGFwZS5sZW5ndGggPD0gMSxcbiAgICAgICAgICAoKSA9PiBgRXJyb3IgaW4gZnVzZWQgY29udjJkOiBvbmx5IHN1cHBvcnRzIHNjYWxhciBvciAxLUQgVGVuc29yIGAgK1xuICAgICAgICAgICAgICBgYmlhcyBmb3IgTkNIVyBmb3JtYXQgYnV0IGdvdCB0aGUgYmlhcyBvZiBgICtcbiAgICAgICAgICAgICAgYHJhbmstJHskYmlhcy5zaGFwZS5sZW5ndGh9LmApO1xuXG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICAkYmlhcy5zaGFwZS5sZW5ndGggPT09IDAgfHwgJGJpYXMuc2hhcGVbMF0gPT09IGNvbnZJbmZvLm91dENoYW5uZWxzIHx8XG4gICAgICAgICAgICAgICRiaWFzLnNoYXBlWzBdID09PSAxLFxuICAgICAgICAgICgpID0+IGBFcnJvciBpbiBmdXNlZCBjb252MmQ6IGJpYXMgc2hhcGUgKCR7JGJpYXMuc2hhcGV9KSBpcyBub3QgYCArXG4gICAgICAgICAgICAgIGBjb21wYXRpYmxlIHdpdGggdGhlIG51bWJlciBvZiBvdXRwdXQgY2hhbm5lbHMgYCArXG4gICAgICAgICAgICAgIGAoJHtjb252SW5mby5vdXRDaGFubmVsc30pYCk7XG4gICAgfVxuICB9XG5cbiAgbGV0ICRwcmVsdUFjdGl2YXRpb25XZWlnaHRzOiBUZW5zb3I7XG4gIGlmIChwcmVsdUFjdGl2YXRpb25XZWlnaHRzICE9IG51bGwpIHtcbiAgICAvLyBQUmVMVSdzIGFjdGl2YXRpb24gd2VpZ2h0cyBjb3VsZCBiZSBhIHNjYWxhciwgYSAxLUQgdGVuc29yIG9yIGEgMy1EXG4gICAgLy8gdGVuc29yLlxuICAgIGNvbnN0IGFscGhhU2hhcGUgPSBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLnNoYXBlO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBhbHBoYVNoYXBlLmxlbmd0aCA8PSAxIHx8IGFscGhhU2hhcGUubGVuZ3RoID09PSAzLFxuICAgICAgICAoKSA9PiBgRXJyb3IgaW4gZnVzZWQgY29udjJkOiBvbmx5IHN1cHBvcnRzIHNjYWxhciwgMS1EIFRlbnNvciBvciBgICtcbiAgICAgICAgICAgIGAzLUQgVGVuc29yIFBSZUxVIGFjdGl2YXRpb24gd2VpZ2h0cyBidXQgZ290IGEgdGVuc29yIG9mIGAgK1xuICAgICAgICAgICAgYHJhbmstJHthbHBoYVNoYXBlLmxlbmd0aH0uYCk7XG5cbiAgICBpZiAoYWxwaGFTaGFwZS5sZW5ndGggPT09IDEpIHtcbiAgICAgIC8vIFdoZXRoZXIgdGhlIGRhdGEgZm9ybWF0IGlzIE5DSFcgb3IgTkhXQywgdGhlIDEtRCBQUmVMVSBhY3RpdmF0aW9uXG4gICAgICAvLyB3ZWlnaHRzIHRlbnNvciBzaG91bGQgYmUgYWxpZ25lZCB3aXRoIHRoZSBvdXRwdXQgY2hhbm5lbHMgb2YgY29udjJkXG4gICAgICAvLyByZXN1bHQuXG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBhbHBoYVNoYXBlWzBdID09PSAxIHx8IGFscGhhU2hhcGVbMF0gPT09IGNvbnZJbmZvLm91dENoYW5uZWxzLFxuICAgICAgICAgICgpID0+IGBFcnJvciBpbiBmdXNlZCBjb252MmQ6IFBSZUxVIGFjdGl2YXRpb24gd2VpZ2h0cyBgICtcbiAgICAgICAgICAgICAgYCgke2FscGhhU2hhcGV9KSBpcyBub3QgY29tcGF0aWJsZSB3aXRoIHRoZSBudW1iZXIgb2Ygb3V0cHV0IGAgK1xuICAgICAgICAgICAgICBgY2hhbm5lbHMgKCR7Y29udkluZm8ub3V0Q2hhbm5lbHN9KS5gKTtcbiAgICB9IGVsc2UgaWYgKGFscGhhU2hhcGUubGVuZ3RoID09PSAzKSB7XG4gICAgICAvLyBXaGV0aGVyIHRoZSBkYXRhIGZvcm1hdCBpcyBOQ0hXIG9yIE5IV0MsIHRoZSBQUmVMVSBhY3RpdmF0aW9uIHdlaWdodHNcbiAgICAgIC8vIHRlbnNvciBzaG91bGQgaGFzIHRoZSBjb21wYXRpYmxlIHNoYXBlIHdpdGggdGhlIHJlc3VsdCBvZiBjb252MmQuXG4gICAgICB0cnkge1xuICAgICAgICBicm9hZGNhc3RfdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RTaGFwZShcbiAgICAgICAgICAgIGFscGhhU2hhcGUsIGNvbnZJbmZvLm91dFNoYXBlKTtcbiAgICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgICAgY29uc3QgZXJyTXNnID1cbiAgICAgICAgICAgIGBFcnJvciBpbiBmdXNlZCBjb252MmQ6IFBSZUxVIGFjdGl2YXRpb24gd2VpZ2h0cyAoJHthbHBoYVNoYXBlfSkgYCArXG4gICAgICAgICAgICBgaXMgbm90IGNvbXBhdGlibGUgd2l0aCB0aGUgb3V0cHV0IHNoYXBlIG9mIHRoZSBjb252MmQgYCArXG4gICAgICAgICAgICBgKCR7Y29udkluZm8ub3V0U2hhcGV9KS5gO1xuICAgICAgICB0aHJvdyBFcnJvcihlcnJNc2cpO1xuICAgICAgfVxuICAgIH1cblxuICAgICRwcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gY29udmVydFRvVGVuc29yKFxuICAgICAgICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLCAncHJlbHUgd2VpZ2h0cycsICdmdXNlZCBjb252MmQnKTtcbiAgfVxuXG4gIGNvbnN0IGdyYWQgPSAoZHk6IFRlbnNvcjRELCBzYXZlZDogVGVuc29yW10pID0+IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZGF0YUZvcm1hdCA9PT0gJ05IV0MnLFxuICAgICAgICAoKSA9PiBgRXJyb3IgaW4gZ3JhZGllbnQgb2YgZnVzZWQgY29udjJEOiBnb3QgZGF0YUZvcm1hdCBvZiAke1xuICAgICAgICAgICAgZGF0YUZvcm1hdH0gYnV0IG9ubHkgTkhXQyBpcyBjdXJyZW50bHkgc3VwcG9ydGVkLmApO1xuXG4gICAgY29uc3QgWyRmaWx0ZXIsIHg0RCwgeSwgJGJpYXNdID1cbiAgICAgICAgc2F2ZWQgYXMgW1RlbnNvcjRELCBUZW5zb3I0RCwgVGVuc29yNEQsIFRlbnNvcl07XG5cbiAgICBjb25zdCBkeUFjdGl2YXRpb24gPSBnZXRGdXNlZER5QWN0aXZhdGlvbihkeSwgeSwgYWN0aXZhdGlvbikgYXMgVGVuc29yNEQ7XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgY29udl91dGlsLnR1cGxlVmFsdWVzQXJlT25lKGRpbGF0aW9ucyksXG4gICAgICAgICgpID0+ICdFcnJvciBpbiBncmFkaWVudCBvZiBmdXNlZCBjb252MkQ6ICcgK1xuICAgICAgICAgICAgYGRpbGF0aW9uIHJhdGVzIGdyZWF0ZXIgdGhhbiAxIGAgK1xuICAgICAgICAgICAgYGFyZSBub3QgeWV0IHN1cHBvcnRlZCBpbiBncmFkaWVudHMuIEdvdCBkaWxhdGlvbnMgJyR7ZGlsYXRpb25zfSdgKTtcblxuICAgIGNvbnN0IHhEZXIgPVxuICAgICAgICBjb252MkRCYWNrcHJvcElucHV0KHg0RC5zaGFwZSwgZHlBY3RpdmF0aW9uLCAkZmlsdGVyLCBzdHJpZGVzLCBwYWQpO1xuICAgIGNvbnN0IGZpbHRlckRlciA9XG4gICAgICAgIGNvbnYyREJhY2twcm9wRmlsdGVyKHg0RCwgZHlBY3RpdmF0aW9uLCAkZmlsdGVyLnNoYXBlLCBzdHJpZGVzLCBwYWQpO1xuICAgIGNvbnN0IGRlcjogVGVuc29yW10gPSBbeERlciwgZmlsdGVyRGVyXTtcblxuICAgIGlmICgkYmlhcyAhPSBudWxsKSB7XG4gICAgICBjb25zdCBiaWFzRGVyID0gZ2V0RnVzZWRCaWFzR3JhZGllbnQoJGJpYXMsIGR5QWN0aXZhdGlvbik7XG4gICAgICBkZXIucHVzaChiaWFzRGVyKTtcbiAgICB9XG4gICAgcmV0dXJuIGRlcjtcbiAgfTtcblxuICBjb25zdCBpbnB1dHM6IEZ1c2VkQ29udjJESW5wdXRzID0ge1xuICAgIHg6IHg0RCxcbiAgICBmaWx0ZXI6ICRmaWx0ZXIsXG4gICAgYmlhczogJGJpYXMsXG4gICAgcHJlbHVBY3RpdmF0aW9uV2VpZ2h0czogJHByZWx1QWN0aXZhdGlvbldlaWdodHNcbiAgfTtcblxuICBjb25zdCBhdHRyczogRnVzZWRDb252MkRBdHRycyA9IHtcbiAgICBzdHJpZGVzLFxuICAgIHBhZCxcbiAgICBkYXRhRm9ybWF0LFxuICAgIGRpbGF0aW9ucyxcbiAgICBkaW1Sb3VuZGluZ01vZGUsXG4gICAgYWN0aXZhdGlvbixcbiAgICBsZWFreXJlbHVBbHBoYVxuICB9O1xuXG4gIC8vIERlcGVuZGluZyBvbiB0aGUgdGhlIHBhcmFtcyBwYXNzZWQgaW4gd2Ugd2lsbCBoYXZlIGRpZmZlcmVudCBudW1iZXIgb2ZcbiAgLy8gaW5wdXRzIGFuZCB0aHVzIGEgYSBkaWZmZXJlbnQgbnVtYmVyIG9mIGVsZW1lbnRzIGluIHRoZSBncmFkaWVudC5cbiAgaWYgKGJpYXMgPT0gbnVsbCkge1xuICAgIGNvbnN0IGN1c3RvbU9wID1cbiAgICAgICAgY3VzdG9tR3JhZCgoeDREOiBUZW5zb3I0RCwgZmlsdGVyOiBUZW5zb3I0RCwgc2F2ZTogR3JhZFNhdmVGdW5jKSA9PiB7XG4gICAgICAgICAgbGV0IHJlczogVGVuc29yNER8VGVuc29yM0QgPVxuICAgICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gICAgICAgICAgICAgIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgICAgICBGdXNlZENvbnYyRCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG5cbiAgICAgICAgICBzYXZlKFtmaWx0ZXIsIHg0RCwgcmVzXSk7XG5cbiAgICAgICAgICBpZiAocmVzaGFwZWRUbzREKSB7XG4gICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gICAgICAgICAgICByZXMgPSByZXNoYXBlKHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM11dKSBhc1xuICAgICAgICAgICAgICAgIFRlbnNvcjNEO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIHJldHVybiB7dmFsdWU6IHJlcywgZ3JhZEZ1bmM6IGdyYWR9O1xuICAgICAgICB9KTtcbiAgICByZXR1cm4gY3VzdG9tT3AoeDRELCAkZmlsdGVyKSBhcyBUO1xuICB9IGVsc2Uge1xuICAgIGNvbnN0IGN1c3RvbU9wV2l0aEJpYXMgPSBjdXN0b21HcmFkKFxuICAgICAgICAoeDREOiBUZW5zb3I0RCwgZmlsdGVyOiBUZW5zb3I0RCwgYmlhczogVGVuc29yLCBzYXZlOiBHcmFkU2F2ZUZ1bmMpID0+IHtcbiAgICAgICAgICBsZXQgcmVzOiBUZW5zb3I0RHxUZW5zb3IzRCA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgIEZ1c2VkQ29udjJELCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xuXG4gICAgICAgICAgc2F2ZShbZmlsdGVyLCB4NEQsIHJlcywgYmlhc10pO1xuXG4gICAgICAgICAgaWYgKHJlc2hhcGVkVG80RCkge1xuICAgICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICAgICAgICAgICAgcmVzID0gcmVzaGFwZShyZXMsIFtyZXMuc2hhcGVbMV0sIHJlcy5zaGFwZVsyXSwgcmVzLnNoYXBlWzNdXSkgYXNcbiAgICAgICAgICAgICAgICBUZW5zb3IzRDtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICByZXR1cm4ge3ZhbHVlOiByZXMsIGdyYWRGdW5jOiBncmFkfTtcbiAgICAgICAgfSk7XG5cbiAgICByZXR1cm4gY3VzdG9tT3BXaXRoQmlhcyh4NEQsICRmaWx0ZXIsICRiaWFzKSBhcyBUO1xuICB9XG59XG5leHBvcnQgY29uc3QgY29udjJkID0gLyogQF9fUFVSRV9fICovIG9wKHtmdXNlZENvbnYyZF99KTtcbiJdfQ==