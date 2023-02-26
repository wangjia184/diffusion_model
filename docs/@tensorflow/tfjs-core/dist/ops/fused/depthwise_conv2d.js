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
import { FusedDepthwiseConv2D } from '../../kernel_names';
import { makeTypesMatch } from '../../tensor_util';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { add } from '../add';
import * as broadcast_util from '../broadcast_util';
import * as conv_util from '../conv_util';
import { depthwiseConv2d as unfusedDepthwiseConv2d } from '../depthwise_conv2d';
import { depthwiseConv2dNativeBackpropFilter } from '../depthwise_conv2d_native_backprop_filter';
import { depthwiseConv2dNativeBackpropInput } from '../depthwise_conv2d_native_backprop_input';
import { applyActivation, getFusedBiasGradient, getFusedDyActivation, shouldFuse } from '../fused_util';
import { op } from '../operation';
import { reshape } from '../reshape';
/**
 * Computes depthwise 2D convolution, optionally fused with adding a
 * bias and applying an activation.
 *
 * Given a 4D `input` array and a `filter` array of shape
 * `[filterHeight, filterWidth, inChannels, channelMultiplier]` containing
 * `inChannels` convolutional filters of depth 1, this op applies a
 * different filter to each input channel (expanding from 1 channel to
 * `channelMultiplier` channels for each), then concatenates the results
 * together. The output has `inChannels * channelMultiplier` channels.
 *
 * See
 * [https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d](
 *     https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d)
 * for more details.
 *
 * @param obj An object with the following properties:
 * @param x The input tensor, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter tensor, rank 4, of shape
 *     `[filterHeight, filterWidth, inChannels, channelMultiplier]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`. If strides is a single number, then `strideHeight ==
 * strideWidth`.
 * @param pad The type of padding algorithm.
 *   - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *   - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 *   - For more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dilations The dilation rates: `[dilationHeight, dilationWidth]`
 *     in which we sample input values across the height and width dimensions
 *     in atrous convolution. Defaults to `[1, 1]`. If `rate` is a single
 *     number, then `dilationHeight == dilationWidth`. If it is greater than
 *     1, then all values of `strides` must be 1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels]. Only "NHWC" is currently supported.
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 * @param bias Tensor to be added to the result.
 * @param activation Name of activation kernel (defaults to `linear`).
 * @param preluActivationWeights Tensor of prelu weights to be applied as part
 *     of a `prelu` activation, typically the same shape as `x`.
 * @param leakyreluAlpha Optional. Alpha to be applied as part of a `leakyrelu`
 *     activation.
 */
function fusedDepthwiseConv2d_({ x, filter, strides, pad, dataFormat = 'NHWC', dilations = [1, 1], dimRoundingMode, bias, activation = 'linear', preluActivationWeights, leakyreluAlpha }) {
    if (shouldFuse(ENGINE.state.gradientDepth, activation) === false) {
        let result = unfusedDepthwiseConv2d(x, filter, strides, pad, dataFormat, dilations, dimRoundingMode);
        if (bias != null) {
            result = add(result, bias);
        }
        return applyActivation(result, activation, preluActivationWeights, leakyreluAlpha);
    }
    const $x = convertToTensor(x, 'x', 'depthwiseConv2d', 'float32');
    const $filter = convertToTensor(filter, 'filter', 'depthwiseConv2d', 'float32');
    let x4D = $x;
    let reshapedTo4D = false;
    if ($x.rank === 3) {
        reshapedTo4D = true;
        x4D = reshape($x, [1, $x.shape[0], $x.shape[1], $x.shape[2]]);
    }
    util.assert(x4D.rank === 4, () => `Error in fused depthwiseConv2d: input must be rank 4, but got ` +
        `rank ${x4D.rank}.`);
    util.assert($filter.rank === 4, () => `Error in fused depthwiseConv2d: filter must be rank 4, ` +
        `but got rank ${$filter.rank}.`);
    util.assert(x4D.shape[3] === $filter.shape[2], () => `Error in fused depthwiseConv2d: number of input channels ` +
        `(${x4D.shape[3]}) must match the inChannels dimension in ` +
        `filter ${$filter.shape[2]}.`);
    if (dilations == null) {
        dilations = [1, 1];
    }
    util.assert(conv_util.eitherStridesOrDilationsAreOne(strides, dilations), () => 'Error in fused depthwiseConv2d: Either strides or dilations must ' +
        `be 1. Got strides ${strides} and dilations '${dilations}'`);
    conv_util.checkPadOnDimRoundingMode('fused depthwiseConv2d', pad, dimRoundingMode);
    const convInfo = conv_util.computeConv2DInfo(x4D.shape, $filter.shape, strides, dilations, pad, dimRoundingMode, true /* depthwise */);
    let $bias;
    if (bias != null) {
        $bias = convertToTensor(bias, 'bias', 'fused conv2d');
        [$bias] = makeTypesMatch($bias, $x);
        broadcast_util.assertAndGetBroadcastShape(convInfo.outShape, $bias.shape);
    }
    let $preluActivationWeights;
    if (preluActivationWeights != null) {
        $preluActivationWeights = convertToTensor(preluActivationWeights, 'prelu weights', 'fused depthwiseConv2d');
    }
    const grad = (dy, saved) => {
        util.assert(conv_util.tupleValuesAreOne(dilations), () => 'Error in gradient of fused depthwiseConv2d: dilation rates ' +
            `greater than 1 are not yet supported. Got dilations ` +
            `'${dilations}'`);
        const [$filter, x4D, y, bias] = saved;
        const dyActivation = getFusedDyActivation(dy, y, activation);
        const xDer = depthwiseConv2dNativeBackpropInput(x4D.shape, dyActivation, $filter, strides, pad, dilations, dimRoundingMode);
        const filterDer = depthwiseConv2dNativeBackpropFilter(x4D, dyActivation, $filter.shape, strides, pad, dilations, dimRoundingMode);
        if (bias != null) {
            const biasDer = getFusedBiasGradient($bias, dyActivation);
            return [xDer, filterDer, biasDer];
        }
        return [xDer, filterDer];
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
            // tslint:disable-next-line: no-unnecessary-type-assertion
            let res = ENGINE.runKernel(FusedDepthwiseConv2D, inputs, attrs);
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
            // tslint:disable-next-line: no-unnecessary-type-assertion
            let res = ENGINE.runKernel(FusedDepthwiseConv2D, inputs, attrs);
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
export const depthwiseConv2d = /* @__PURE__ */ op({ fusedDepthwiseConv2d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVwdGh3aXNlX2NvbnYyZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2Z1c2VkL2RlcHRod2lzZV9jb252MmQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNwQyxPQUFPLEVBQUMsVUFBVSxFQUFDLE1BQU0saUJBQWlCLENBQUM7QUFDM0MsT0FBTyxFQUFDLG9CQUFvQixFQUF3RCxNQUFNLG9CQUFvQixDQUFDO0FBSS9HLE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxtQkFBbUIsQ0FBQztBQUNqRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxLQUFLLElBQUksTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUMzQixPQUFPLEtBQUssY0FBYyxNQUFNLG1CQUFtQixDQUFDO0FBQ3BELE9BQU8sS0FBSyxTQUFTLE1BQU0sY0FBYyxDQUFDO0FBQzFDLE9BQU8sRUFBQyxlQUFlLElBQUksc0JBQXNCLEVBQUMsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RSxPQUFPLEVBQUMsbUNBQW1DLEVBQUMsTUFBTSw0Q0FBNEMsQ0FBQztBQUMvRixPQUFPLEVBQUMsa0NBQWtDLEVBQUMsTUFBTSwyQ0FBMkMsQ0FBQztBQUU3RixPQUFPLEVBQUMsZUFBZSxFQUFFLG9CQUFvQixFQUFFLG9CQUFvQixFQUFFLFVBQVUsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUN0RyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFFbkM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7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O0FBQ0QsTUFBTSxDQUFDLE1BQU0sZUFBZSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxxQkFBcUIsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtjdXN0b21HcmFkfSBmcm9tICcuLi8uLi9ncmFkaWVudHMnO1xuaW1wb3J0IHtGdXNlZERlcHRod2lzZUNvbnYyRCwgRnVzZWREZXB0aHdpc2VDb252MkRBdHRycywgRnVzZWREZXB0aHdpc2VDb252MkRJbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4vLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yLCBUZW5zb3IzRCwgVGVuc29yNER9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge0dyYWRTYXZlRnVuYywgTmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uLy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge21ha2VUeXBlc01hdGNofSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbCc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi8uLi91dGlsJztcbmltcG9ydCB7YWRkfSBmcm9tICcuLi9hZGQnO1xuaW1wb3J0ICogYXMgYnJvYWRjYXN0X3V0aWwgZnJvbSAnLi4vYnJvYWRjYXN0X3V0aWwnO1xuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uL2NvbnZfdXRpbCc7XG5pbXBvcnQge2RlcHRod2lzZUNvbnYyZCBhcyB1bmZ1c2VkRGVwdGh3aXNlQ29udjJkfSBmcm9tICcuLi9kZXB0aHdpc2VfY29udjJkJztcbmltcG9ydCB7ZGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BGaWx0ZXJ9IGZyb20gJy4uL2RlcHRod2lzZV9jb252MmRfbmF0aXZlX2JhY2twcm9wX2ZpbHRlcic7XG5pbXBvcnQge2RlcHRod2lzZUNvbnYyZE5hdGl2ZUJhY2twcm9wSW5wdXR9IGZyb20gJy4uL2RlcHRod2lzZV9jb252MmRfbmF0aXZlX2JhY2twcm9wX2lucHV0JztcbmltcG9ydCB7QWN0aXZhdGlvbn0gZnJvbSAnLi4vZnVzZWRfdHlwZXMnO1xuaW1wb3J0IHthcHBseUFjdGl2YXRpb24sIGdldEZ1c2VkQmlhc0dyYWRpZW50LCBnZXRGdXNlZER5QWN0aXZhdGlvbiwgc2hvdWxkRnVzZX0gZnJvbSAnLi4vZnVzZWRfdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuLi9yZXNoYXBlJztcblxuLyoqXG4gKiBDb21wdXRlcyBkZXB0aHdpc2UgMkQgY29udm9sdXRpb24sIG9wdGlvbmFsbHkgZnVzZWQgd2l0aCBhZGRpbmcgYVxuICogYmlhcyBhbmQgYXBwbHlpbmcgYW4gYWN0aXZhdGlvbi5cbiAqXG4gKiBHaXZlbiBhIDREIGBpbnB1dGAgYXJyYXkgYW5kIGEgYGZpbHRlcmAgYXJyYXkgb2Ygc2hhcGVcbiAqIGBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgaW5DaGFubmVscywgY2hhbm5lbE11bHRpcGxpZXJdYCBjb250YWluaW5nXG4gKiBgaW5DaGFubmVsc2AgY29udm9sdXRpb25hbCBmaWx0ZXJzIG9mIGRlcHRoIDEsIHRoaXMgb3AgYXBwbGllcyBhXG4gKiBkaWZmZXJlbnQgZmlsdGVyIHRvIGVhY2ggaW5wdXQgY2hhbm5lbCAoZXhwYW5kaW5nIGZyb20gMSBjaGFubmVsIHRvXG4gKiBgY2hhbm5lbE11bHRpcGxpZXJgIGNoYW5uZWxzIGZvciBlYWNoKSwgdGhlbiBjb25jYXRlbmF0ZXMgdGhlIHJlc3VsdHNcbiAqIHRvZ2V0aGVyLiBUaGUgb3V0cHV0IGhhcyBgaW5DaGFubmVscyAqIGNoYW5uZWxNdWx0aXBsaWVyYCBjaGFubmVscy5cbiAqXG4gKiBTZWVcbiAqIFtodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vZGVwdGh3aXNlX2NvbnYyZF0oXG4gKiAgICAgaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2RlcHRod2lzZV9jb252MmQpXG4gKiBmb3IgbW9yZSBkZXRhaWxzLlxuICpcbiAqIEBwYXJhbSBvYmogQW4gb2JqZWN0IHdpdGggdGhlIGZvbGxvd2luZyBwcm9wZXJ0aWVzOlxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvciwgb2YgcmFuayA0IG9yIHJhbmsgMywgb2Ygc2hhcGVcbiAqICAgICBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBpbkNoYW5uZWxzXWAuIElmIHJhbmsgMywgYmF0Y2ggb2YgMSBpc1xuICogYXNzdW1lZC5cbiAqIEBwYXJhbSBmaWx0ZXIgVGhlIGZpbHRlciB0ZW5zb3IsIHJhbmsgNCwgb2Ygc2hhcGVcbiAqICAgICBgW2ZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGgsIGluQ2hhbm5lbHMsIGNoYW5uZWxNdWx0aXBsaWVyXWAuXG4gKiBAcGFyYW0gc3RyaWRlcyBUaGUgc3RyaWRlcyBvZiB0aGUgY29udm9sdXRpb246IGBbc3RyaWRlSGVpZ2h0LFxuICogc3RyaWRlV2lkdGhdYC4gSWYgc3RyaWRlcyBpcyBhIHNpbmdsZSBudW1iZXIsIHRoZW4gYHN0cmlkZUhlaWdodCA9PVxuICogc3RyaWRlV2lkdGhgLlxuICogQHBhcmFtIHBhZCBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobS5cbiAqICAgLSBgc2FtZWAgYW5kIHN0cmlkZSAxOiBvdXRwdXQgd2lsbCBiZSBvZiBzYW1lIHNpemUgYXMgaW5wdXQsXG4gKiAgICAgICByZWdhcmRsZXNzIG9mIGZpbHRlciBzaXplLlxuICogICAtIGB2YWxpZGA6IG91dHB1dCB3aWxsIGJlIHNtYWxsZXIgdGhhbiBpbnB1dCBpZiBmaWx0ZXIgaXMgbGFyZ2VyXG4gKiAgICAgICB0aGFuIDF4MS5cbiAqICAgLSBGb3IgbW9yZSBpbmZvLCBzZWUgdGhpcyBndWlkZTpcbiAqICAgICBbaHR0cHM6Ly93d3cudGVuc29yZmxvdy5vcmcvYXBpX2RvY3MvcHl0aG9uL3RmL25uL2NvbnZvbHV0aW9uXShcbiAqICAgICAgICAgIGh0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9jb252b2x1dGlvbilcbiAqIEBwYXJhbSBkaWxhdGlvbnMgVGhlIGRpbGF0aW9uIHJhdGVzOiBgW2RpbGF0aW9uSGVpZ2h0LCBkaWxhdGlvbldpZHRoXWBcbiAqICAgICBpbiB3aGljaCB3ZSBzYW1wbGUgaW5wdXQgdmFsdWVzIGFjcm9zcyB0aGUgaGVpZ2h0IGFuZCB3aWR0aCBkaW1lbnNpb25zXG4gKiAgICAgaW4gYXRyb3VzIGNvbnZvbHV0aW9uLiBEZWZhdWx0cyB0byBgWzEsIDFdYC4gSWYgYHJhdGVgIGlzIGEgc2luZ2xlXG4gKiAgICAgbnVtYmVyLCB0aGVuIGBkaWxhdGlvbkhlaWdodCA9PSBkaWxhdGlvbldpZHRoYC4gSWYgaXQgaXMgZ3JlYXRlciB0aGFuXG4gKiAgICAgMSwgdGhlbiBhbGwgdmFsdWVzIG9mIGBzdHJpZGVzYCBtdXN0IGJlIDEuXG4gKiBAcGFyYW0gZGF0YUZvcm1hdDogQW4gb3B0aW9uYWwgc3RyaW5nIGZyb206IFwiTkhXQ1wiLCBcIk5DSFdcIi4gRGVmYXVsdHMgdG9cbiAqICAgICBcIk5IV0NcIi4gU3BlY2lmeSB0aGUgZGF0YSBmb3JtYXQgb2YgdGhlIGlucHV0IGFuZCBvdXRwdXQgZGF0YS4gV2l0aCB0aGVcbiAqICAgICBkZWZhdWx0IGZvcm1hdCBcIk5IV0NcIiwgdGhlIGRhdGEgaXMgc3RvcmVkIGluIHRoZSBvcmRlciBvZjogW2JhdGNoLFxuICogICAgIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzXS4gT25seSBcIk5IV0NcIiBpcyBjdXJyZW50bHkgc3VwcG9ydGVkLlxuICogQHBhcmFtIGRpbVJvdW5kaW5nTW9kZSBBIHN0cmluZyBmcm9tOiAnY2VpbCcsICdyb3VuZCcsICdmbG9vcicuIElmIG5vbmUgaXNcbiAqICAgICBwcm92aWRlZCwgaXQgd2lsbCBkZWZhdWx0IHRvIHRydW5jYXRlLlxuICogQHBhcmFtIGJpYXMgVGVuc29yIHRvIGJlIGFkZGVkIHRvIHRoZSByZXN1bHQuXG4gKiBAcGFyYW0gYWN0aXZhdGlvbiBOYW1lIG9mIGFjdGl2YXRpb24ga2VybmVsIChkZWZhdWx0cyB0byBgbGluZWFyYCkuXG4gKiBAcGFyYW0gcHJlbHVBY3RpdmF0aW9uV2VpZ2h0cyBUZW5zb3Igb2YgcHJlbHUgd2VpZ2h0cyB0byBiZSBhcHBsaWVkIGFzIHBhcnRcbiAqICAgICBvZiBhIGBwcmVsdWAgYWN0aXZhdGlvbiwgdHlwaWNhbGx5IHRoZSBzYW1lIHNoYXBlIGFzIGB4YC5cbiAqIEBwYXJhbSBsZWFreXJlbHVBbHBoYSBPcHRpb25hbC4gQWxwaGEgdG8gYmUgYXBwbGllZCBhcyBwYXJ0IG9mIGEgYGxlYWt5cmVsdWBcbiAqICAgICBhY3RpdmF0aW9uLlxuICovXG5mdW5jdGlvbiBmdXNlZERlcHRod2lzZUNvbnYyZF88VCBleHRlbmRzIFRlbnNvcjNEfFRlbnNvcjREPih7XG4gIHgsXG4gIGZpbHRlcixcbiAgc3RyaWRlcyxcbiAgcGFkLFxuICBkYXRhRm9ybWF0ID0gJ05IV0MnLFxuICBkaWxhdGlvbnMgPSBbMSwgMV0sXG4gIGRpbVJvdW5kaW5nTW9kZSxcbiAgYmlhcyxcbiAgYWN0aXZhdGlvbiA9ICdsaW5lYXInLFxuICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLFxuICBsZWFreXJlbHVBbHBoYVxufToge1xuICB4OiBUfFRlbnNvckxpa2UsXG4gIGZpbHRlcjogVGVuc29yNER8VGVuc29yTGlrZSxcbiAgc3RyaWRlczogW251bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gIHBhZDogJ3ZhbGlkJ3wnc2FtZSd8bnVtYmVyLFxuICBkYXRhRm9ybWF0PzogJ05IV0MnfCdOQ0hXJyxcbiAgZGlsYXRpb25zPzogW251bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gIGRpbVJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcsXG4gIGJpYXM/OiBUZW5zb3J8VGVuc29yTGlrZSxcbiAgYWN0aXZhdGlvbj86IEFjdGl2YXRpb24sXG4gIHByZWx1QWN0aXZhdGlvbldlaWdodHM/OiBUZW5zb3IsXG4gIGxlYWt5cmVsdUFscGhhPzogbnVtYmVyXG59KTogVCB7XG4gIGlmIChzaG91bGRGdXNlKEVOR0lORS5zdGF0ZS5ncmFkaWVudERlcHRoLCBhY3RpdmF0aW9uKSA9PT0gZmFsc2UpIHtcbiAgICBsZXQgcmVzdWx0ID0gdW5mdXNlZERlcHRod2lzZUNvbnYyZChcbiAgICAgICAgeCwgZmlsdGVyLCBzdHJpZGVzLCBwYWQsIGRhdGFGb3JtYXQsIGRpbGF0aW9ucywgZGltUm91bmRpbmdNb2RlKTtcbiAgICBpZiAoYmlhcyAhPSBudWxsKSB7XG4gICAgICByZXN1bHQgPSBhZGQocmVzdWx0LCBiaWFzKTtcbiAgICB9XG5cbiAgICByZXR1cm4gYXBwbHlBY3RpdmF0aW9uKFxuICAgICAgICAgICAgICAgcmVzdWx0LCBhY3RpdmF0aW9uLCBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLCBsZWFreXJlbHVBbHBoYSkgYXMgVDtcbiAgfVxuXG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ2RlcHRod2lzZUNvbnYyZCcsICdmbG9hdDMyJyk7XG4gIGNvbnN0ICRmaWx0ZXIgPVxuICAgICAgY29udmVydFRvVGVuc29yKGZpbHRlciwgJ2ZpbHRlcicsICdkZXB0aHdpc2VDb252MmQnLCAnZmxvYXQzMicpO1xuXG4gIGxldCB4NEQgPSAkeCBhcyBUZW5zb3I0RDtcbiAgbGV0IHJlc2hhcGVkVG80RCA9IGZhbHNlO1xuICBpZiAoJHgucmFuayA9PT0gMykge1xuICAgIHJlc2hhcGVkVG80RCA9IHRydWU7XG4gICAgeDREID0gcmVzaGFwZSgkeCwgWzEsICR4LnNoYXBlWzBdLCAkeC5zaGFwZVsxXSwgJHguc2hhcGVbMl1dKTtcbiAgfVxuICB1dGlsLmFzc2VydChcbiAgICAgIHg0RC5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGZ1c2VkIGRlcHRod2lzZUNvbnYyZDogaW5wdXQgbXVzdCBiZSByYW5rIDQsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgYHJhbmsgJHt4NEQucmFua30uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGZpbHRlci5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIGZ1c2VkIGRlcHRod2lzZUNvbnYyZDogZmlsdGVyIG11c3QgYmUgcmFuayA0LCBgICtcbiAgICAgICAgICBgYnV0IGdvdCByYW5rICR7JGZpbHRlci5yYW5rfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4NEQuc2hhcGVbM10gPT09ICRmaWx0ZXIuc2hhcGVbMl0sXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gZnVzZWQgZGVwdGh3aXNlQ29udjJkOiBudW1iZXIgb2YgaW5wdXQgY2hhbm5lbHMgYCArXG4gICAgICAgICAgYCgke3g0RC5zaGFwZVszXX0pIG11c3QgbWF0Y2ggdGhlIGluQ2hhbm5lbHMgZGltZW5zaW9uIGluIGAgK1xuICAgICAgICAgIGBmaWx0ZXIgJHskZmlsdGVyLnNoYXBlWzJdfS5gKTtcbiAgaWYgKGRpbGF0aW9ucyA9PSBudWxsKSB7XG4gICAgZGlsYXRpb25zID0gWzEsIDFdO1xuICB9XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgY29udl91dGlsLmVpdGhlclN0cmlkZXNPckRpbGF0aW9uc0FyZU9uZShzdHJpZGVzLCBkaWxhdGlvbnMpLFxuICAgICAgKCkgPT5cbiAgICAgICAgICAnRXJyb3IgaW4gZnVzZWQgZGVwdGh3aXNlQ29udjJkOiBFaXRoZXIgc3RyaWRlcyBvciBkaWxhdGlvbnMgbXVzdCAnICtcbiAgICAgICAgICBgYmUgMS4gR290IHN0cmlkZXMgJHtzdHJpZGVzfSBhbmQgZGlsYXRpb25zICcke2RpbGF0aW9uc30nYCk7XG4gIGNvbnZfdXRpbC5jaGVja1BhZE9uRGltUm91bmRpbmdNb2RlKFxuICAgICAgJ2Z1c2VkIGRlcHRod2lzZUNvbnYyZCcsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcbiAgY29uc3QgY29udkluZm8gPSBjb252X3V0aWwuY29tcHV0ZUNvbnYyREluZm8oXG4gICAgICB4NEQuc2hhcGUsICRmaWx0ZXIuc2hhcGUsIHN0cmlkZXMsIGRpbGF0aW9ucywgcGFkLCBkaW1Sb3VuZGluZ01vZGUsXG4gICAgICB0cnVlIC8qIGRlcHRod2lzZSAqLyk7XG5cbiAgbGV0ICRiaWFzOiBUZW5zb3I7XG4gIGlmIChiaWFzICE9IG51bGwpIHtcbiAgICAkYmlhcyA9IGNvbnZlcnRUb1RlbnNvcihiaWFzLCAnYmlhcycsICdmdXNlZCBjb252MmQnKTtcbiAgICBbJGJpYXNdID0gbWFrZVR5cGVzTWF0Y2goJGJpYXMsICR4KTtcblxuICAgIGJyb2FkY2FzdF91dGlsLmFzc2VydEFuZEdldEJyb2FkY2FzdFNoYXBlKGNvbnZJbmZvLm91dFNoYXBlLCAkYmlhcy5zaGFwZSk7XG4gIH1cblxuICBsZXQgJHByZWx1QWN0aXZhdGlvbldlaWdodHM6IFRlbnNvcjtcbiAgaWYgKHByZWx1QWN0aXZhdGlvbldlaWdodHMgIT0gbnVsbCkge1xuICAgICRwcmVsdUFjdGl2YXRpb25XZWlnaHRzID0gY29udmVydFRvVGVuc29yKFxuICAgICAgICBwcmVsdUFjdGl2YXRpb25XZWlnaHRzLCAncHJlbHUgd2VpZ2h0cycsICdmdXNlZCBkZXB0aHdpc2VDb252MmQnKTtcbiAgfVxuXG4gIGNvbnN0IGdyYWQgPSAoZHk6IFRlbnNvcjRELCBzYXZlZDogVGVuc29yW10pID0+IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgY29udl91dGlsLnR1cGxlVmFsdWVzQXJlT25lKGRpbGF0aW9ucyksXG4gICAgICAgICgpID0+ICdFcnJvciBpbiBncmFkaWVudCBvZiBmdXNlZCBkZXB0aHdpc2VDb252MmQ6IGRpbGF0aW9uIHJhdGVzICcgK1xuICAgICAgICAgICAgYGdyZWF0ZXIgdGhhbiAxIGFyZSBub3QgeWV0IHN1cHBvcnRlZC4gR290IGRpbGF0aW9ucyBgICtcbiAgICAgICAgICAgIGAnJHtkaWxhdGlvbnN9J2ApO1xuICAgIGNvbnN0IFskZmlsdGVyLCB4NEQsIHksIGJpYXNdID0gc2F2ZWQ7XG5cbiAgICBjb25zdCBkeUFjdGl2YXRpb24gPSBnZXRGdXNlZER5QWN0aXZhdGlvbihkeSwgeSwgYWN0aXZhdGlvbikgYXMgVGVuc29yNEQ7XG5cbiAgICBjb25zdCB4RGVyID0gZGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BJbnB1dChcbiAgICAgICAgKHg0RCBhcyBUZW5zb3I0RCkuc2hhcGUsIGR5QWN0aXZhdGlvbiwgJGZpbHRlciBhcyBUZW5zb3I0RCwgc3RyaWRlcyxcbiAgICAgICAgcGFkLCBkaWxhdGlvbnMsIGRpbVJvdW5kaW5nTW9kZSk7XG4gICAgY29uc3QgZmlsdGVyRGVyID0gZGVwdGh3aXNlQ29udjJkTmF0aXZlQmFja3Byb3BGaWx0ZXIoXG4gICAgICAgIHg0RCBhcyBUZW5zb3I0RCwgZHlBY3RpdmF0aW9uLCAoJGZpbHRlciBhcyBUZW5zb3I0RCkuc2hhcGUsIHN0cmlkZXMsXG4gICAgICAgIHBhZCwgZGlsYXRpb25zLCBkaW1Sb3VuZGluZ01vZGUpO1xuXG4gICAgaWYgKGJpYXMgIT0gbnVsbCkge1xuICAgICAgY29uc3QgYmlhc0RlciA9IGdldEZ1c2VkQmlhc0dyYWRpZW50KCRiaWFzLCBkeUFjdGl2YXRpb24pO1xuICAgICAgcmV0dXJuIFt4RGVyLCBmaWx0ZXJEZXIsIGJpYXNEZXJdO1xuICAgIH1cbiAgICByZXR1cm4gW3hEZXIsIGZpbHRlckRlcl07XG4gIH07XG5cbiAgY29uc3QgaW5wdXRzOiBGdXNlZERlcHRod2lzZUNvbnYyRElucHV0cyA9IHtcbiAgICB4OiB4NEQsXG4gICAgZmlsdGVyOiAkZmlsdGVyLFxuICAgIGJpYXM6ICRiaWFzLFxuICAgIHByZWx1QWN0aXZhdGlvbldlaWdodHM6ICRwcmVsdUFjdGl2YXRpb25XZWlnaHRzXG4gIH07XG4gIGNvbnN0IGF0dHJzOiBGdXNlZERlcHRod2lzZUNvbnYyREF0dHJzID0ge1xuICAgIHN0cmlkZXMsXG4gICAgcGFkLFxuICAgIGRhdGFGb3JtYXQsXG4gICAgZGlsYXRpb25zLFxuICAgIGRpbVJvdW5kaW5nTW9kZSxcbiAgICBhY3RpdmF0aW9uLFxuICAgIGxlYWt5cmVsdUFscGhhXG4gIH07XG5cbiAgLy8gRGVwZW5kaW5nIG9uIHRoZSB0aGUgcGFyYW1zIHBhc3NlZCBpbiB3ZSB3aWxsIGhhdmUgZGlmZmVyZW50IG51bWJlciBvZlxuICAvLyBpbnB1dHMgYW5kIHRodXMgYSBhIGRpZmZlcmVudCBudW1iZXIgb2YgZWxlbWVudHMgaW4gdGhlIGdyYWRpZW50LlxuICBpZiAoYmlhcyA9PSBudWxsKSB7XG4gICAgY29uc3QgY3VzdG9tT3AgPVxuICAgICAgICBjdXN0b21HcmFkKCh4NEQ6IFRlbnNvcjRELCBmaWx0ZXI6IFRlbnNvcjRELCBzYXZlOiBHcmFkU2F2ZUZ1bmMpID0+IHtcbiAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gICAgICAgICAgbGV0IHJlczogVGVuc29yNER8VGVuc29yM0QgPSBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgICBGdXNlZERlcHRod2lzZUNvbnYyRCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICAgICAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKTtcblxuICAgICAgICAgIHNhdmUoW2ZpbHRlciwgeDRELCByZXNdKTtcblxuICAgICAgICAgIGlmIChyZXNoYXBlZFRvNEQpIHtcbiAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgICAgICAgICAgIHJlcyA9IHJlc2hhcGUocmVzLCBbcmVzLnNoYXBlWzFdLCByZXMuc2hhcGVbMl0sIHJlcy5zaGFwZVszXV0pIGFzXG4gICAgICAgICAgICAgICAgVGVuc29yM0Q7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgcmV0dXJuIHt2YWx1ZTogcmVzLCBncmFkRnVuYzogZ3JhZH07XG4gICAgICAgIH0pO1xuICAgIHJldHVybiBjdXN0b21PcCh4NEQsICRmaWx0ZXIpIGFzIFQ7XG4gIH0gZWxzZSB7XG4gICAgY29uc3QgY3VzdG9tT3BXaXRoQmlhcyA9IGN1c3RvbUdyYWQoXG4gICAgICAgICh4NEQ6IFRlbnNvcjRELCBmaWx0ZXI6IFRlbnNvcjRELCBiaWFzOiBUZW5zb3IsIHNhdmU6IEdyYWRTYXZlRnVuYykgPT4ge1xuICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgICAgICAgICBsZXQgcmVzOiBUZW5zb3I0RHxUZW5zb3IzRCA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgIEZ1c2VkRGVwdGh3aXNlQ29udjJELCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xuXG4gICAgICAgICAgc2F2ZShbZmlsdGVyLCB4NEQsIHJlcywgYmlhc10pO1xuXG4gICAgICAgICAgaWYgKHJlc2hhcGVkVG80RCkge1xuICAgICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICAgICAgICAgICAgcmVzID0gcmVzaGFwZShyZXMsIFtyZXMuc2hhcGVbMV0sIHJlcy5zaGFwZVsyXSwgcmVzLnNoYXBlWzNdXSkgYXNcbiAgICAgICAgICAgICAgICBUZW5zb3IzRDtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICByZXR1cm4ge3ZhbHVlOiByZXMsIGdyYWRGdW5jOiBncmFkfTtcbiAgICAgICAgfSk7XG5cbiAgICByZXR1cm4gY3VzdG9tT3BXaXRoQmlhcyh4NEQsICRmaWx0ZXIsICRiaWFzKSBhcyBUO1xuICB9XG59XG5leHBvcnQgY29uc3QgZGVwdGh3aXNlQ29udjJkID0gLyogQF9fUFVSRV9fICovIG9wKHtmdXNlZERlcHRod2lzZUNvbnYyZF99KTtcbiJdfQ==