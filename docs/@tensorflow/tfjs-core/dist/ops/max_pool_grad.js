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
import { ENGINE } from '../engine';
import { MaxPoolGrad } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import * as conv_util from './conv_util';
import { op } from './operation';
/**
 * Computes the backprop of a 2D max pool.
 *
 * @param dy The dy error, of rank 4 or rank 3 of shape
 *     [batchSize, height, width, channels]. If rank 3, batch of 1 is
 * assumed.
 * @param input The original input image, of rank 4, of shape
 *     [batchSize, height, width, channels].
 * @param output The original output image, of rank 4, of shape
 *     [batchSize, outHeight, outWidth, channels].
 * @param filterSize The filter size: `[filterHeight, filterWidth]`. If
 *     `filterSize` is a single number, then `filterHeight == filterWidth`.
 * @param strides The strides of the pooling: `[strideHeight, strideWidth]`. If
 *     `strides` is a single number, then `strideHeight == strideWidth`.
 * @param pad The type of padding algorithm used in the forward prop of the op.
 *     'same', 'valid', for more info, see this guide:
 *     [https://www.tensorflow.org/api_docs/python/tf/nn/convolution](
 *          https://www.tensorflow.org/api_docs/python/tf/nn/convolution)
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function maxPoolGrad_(dy, input, output, filterSize, strides, pad, dimRoundingMode) {
    const $dy = convertToTensor(dy, 'dy', 'maxPoolGrad');
    const $input = convertToTensor(input, 'input', 'maxPoolGrad');
    const $output = convertToTensor(output, 'output', 'maxPoolGrad');
    util.assert($input.rank === $dy.rank, () => `Rank of input (${$input.rank}) does not match rank of dy ` +
        `(${$dy.rank})`);
    util.assert($dy.rank === 4, () => `Error in maxPoolGrad: dy must be rank 4 but got rank ` +
        `${$dy.rank}.`);
    util.assert($input.rank === 4, () => `Error in maxPoolGrad: input must be rank 4 but got rank ` +
        `${$input.rank}.`);
    conv_util.checkPadOnDimRoundingMode('maxPoolGrad', pad, dimRoundingMode);
    const inputs = { dy: $dy, input: $input, output: $output };
    const attrs = { filterSize, strides, pad, dimRoundingMode };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    return ENGINE.runKernel(MaxPoolGrad, inputs, attrs);
}
export const maxPoolGrad = /* @__PURE__ */ op({ maxPoolGrad_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWF4X3Bvb2xfZ3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL21heF9wb29sX2dyYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsV0FBVyxFQUFzQyxNQUFNLGlCQUFpQixDQUFDO0FBSWpGLE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUVuRCxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxPQUFPLEtBQUssU0FBUyxNQUFNLGFBQWEsQ0FBQztBQUN6QyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQW9CRztBQUNILFNBQVMsWUFBWSxDQUNqQixFQUF1QixFQUFFLEtBQTBCLEVBQ25ELE1BQTJCLEVBQUUsVUFBbUMsRUFDaEUsT0FBZ0MsRUFDaEMsR0FBb0QsRUFDcEQsZUFBd0M7SUFDMUMsTUFBTSxHQUFHLEdBQUcsZUFBZSxDQUFDLEVBQUUsRUFBRSxJQUFJLEVBQUUsYUFBYSxDQUFDLENBQUM7SUFDckQsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsYUFBYSxDQUFDLENBQUM7SUFDOUQsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsYUFBYSxDQUFDLENBQUM7SUFFakUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLENBQUMsSUFBSSxLQUFLLEdBQUcsQ0FBQyxJQUFJLEVBQ3hCLEdBQUcsRUFBRSxDQUFDLGtCQUFrQixNQUFNLENBQUMsSUFBSSw4QkFBOEI7UUFDN0QsSUFBSSxHQUFHLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUV6QixJQUFJLENBQUMsTUFBTSxDQUNQLEdBQUcsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNkLEdBQUcsRUFBRSxDQUFDLHVEQUF1RDtRQUN6RCxHQUFHLEdBQUcsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQ3hCLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLEdBQUcsRUFBRSxDQUFDLDBEQUEwRDtRQUM1RCxHQUFHLE1BQU0sQ0FBQyxJQUFJLEdBQUcsQ0FBQyxDQUFDO0lBQzNCLFNBQVMsQ0FBQyx5QkFBeUIsQ0FBQyxhQUFhLEVBQUUsR0FBRyxFQUFFLGVBQWUsQ0FBQyxDQUFDO0lBQ3pFLE1BQU0sTUFBTSxHQUFzQixFQUFDLEVBQUUsRUFBRSxHQUFHLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxNQUFNLEVBQUUsT0FBTyxFQUFDLENBQUM7SUFDNUUsTUFBTSxLQUFLLEdBQXFCLEVBQUMsVUFBVSxFQUFFLE9BQU8sRUFBRSxHQUFHLEVBQUUsZUFBZSxFQUFDLENBQUM7SUFFNUUsMERBQTBEO0lBQzFELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDWixXQUFXLEVBQUUsTUFBbUMsRUFDaEQsS0FBZ0MsQ0FBYSxDQUFDO0FBQzNELENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxXQUFXLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLFlBQVksRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtNYXhQb29sR3JhZCwgTWF4UG9vbEdyYWRBdHRycywgTWF4UG9vbEdyYWRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yNER9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuL2NvbnZfdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIGJhY2twcm9wIG9mIGEgMkQgbWF4IHBvb2wuXG4gKlxuICogQHBhcmFtIGR5IFRoZSBkeSBlcnJvciwgb2YgcmFuayA0IG9yIHJhbmsgMyBvZiBzaGFwZVxuICogICAgIFtiYXRjaFNpemUsIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzXS4gSWYgcmFuayAzLCBiYXRjaCBvZiAxIGlzXG4gKiBhc3N1bWVkLlxuICogQHBhcmFtIGlucHV0IFRoZSBvcmlnaW5hbCBpbnB1dCBpbWFnZSwgb2YgcmFuayA0LCBvZiBzaGFwZVxuICogICAgIFtiYXRjaFNpemUsIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzXS5cbiAqIEBwYXJhbSBvdXRwdXQgVGhlIG9yaWdpbmFsIG91dHB1dCBpbWFnZSwgb2YgcmFuayA0LCBvZiBzaGFwZVxuICogICAgIFtiYXRjaFNpemUsIG91dEhlaWdodCwgb3V0V2lkdGgsIGNoYW5uZWxzXS5cbiAqIEBwYXJhbSBmaWx0ZXJTaXplIFRoZSBmaWx0ZXIgc2l6ZTogYFtmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoXWAuIElmXG4gKiAgICAgYGZpbHRlclNpemVgIGlzIGEgc2luZ2xlIG51bWJlciwgdGhlbiBgZmlsdGVySGVpZ2h0ID09IGZpbHRlcldpZHRoYC5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBwb29saW5nOiBgW3N0cmlkZUhlaWdodCwgc3RyaWRlV2lkdGhdYC4gSWZcbiAqICAgICBgc3RyaWRlc2AgaXMgYSBzaW5nbGUgbnVtYmVyLCB0aGVuIGBzdHJpZGVIZWlnaHQgPT0gc3RyaWRlV2lkdGhgLlxuICogQHBhcmFtIHBhZCBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobSB1c2VkIGluIHRoZSBmb3J3YXJkIHByb3Agb2YgdGhlIG9wLlxuICogICAgICdzYW1lJywgJ3ZhbGlkJywgZm9yIG1vcmUgaW5mbywgc2VlIHRoaXMgZ3VpZGU6XG4gKiAgICAgW2h0dHBzOi8vd3d3LnRlbnNvcmZsb3cub3JnL2FwaV9kb2NzL3B5dGhvbi90Zi9ubi9jb252b2x1dGlvbl0oXG4gKiAgICAgICAgICBodHRwczovL3d3dy50ZW5zb3JmbG93Lm9yZy9hcGlfZG9jcy9weXRob24vdGYvbm4vY29udm9sdXRpb24pXG4gKiBAcGFyYW0gZGltUm91bmRpbmdNb2RlIEEgc3RyaW5nIGZyb206ICdjZWlsJywgJ3JvdW5kJywgJ2Zsb29yJy4gSWYgbm9uZSBpc1xuICogICAgIHByb3ZpZGVkLCBpdCB3aWxsIGRlZmF1bHQgdG8gdHJ1bmNhdGUuXG4gKi9cbmZ1bmN0aW9uIG1heFBvb2xHcmFkXyhcbiAgICBkeTogVGVuc29yNER8VGVuc29yTGlrZSwgaW5wdXQ6IFRlbnNvcjREfFRlbnNvckxpa2UsXG4gICAgb3V0cHV0OiBUZW5zb3I0RHxUZW5zb3JMaWtlLCBmaWx0ZXJTaXplOiBbbnVtYmVyLCBudW1iZXJdfG51bWJlcixcbiAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdfG51bWJlcixcbiAgICBwYWQ6ICd2YWxpZCd8J3NhbWUnfG51bWJlcnxjb252X3V0aWwuRXhwbGljaXRQYWRkaW5nLFxuICAgIGRpbVJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcpOiBUZW5zb3I0RCB7XG4gIGNvbnN0ICRkeSA9IGNvbnZlcnRUb1RlbnNvcihkeSwgJ2R5JywgJ21heFBvb2xHcmFkJyk7XG4gIGNvbnN0ICRpbnB1dCA9IGNvbnZlcnRUb1RlbnNvcihpbnB1dCwgJ2lucHV0JywgJ21heFBvb2xHcmFkJyk7XG4gIGNvbnN0ICRvdXRwdXQgPSBjb252ZXJ0VG9UZW5zb3Iob3V0cHV0LCAnb3V0cHV0JywgJ21heFBvb2xHcmFkJyk7XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkaW5wdXQucmFuayA9PT0gJGR5LnJhbmssXG4gICAgICAoKSA9PiBgUmFuayBvZiBpbnB1dCAoJHskaW5wdXQucmFua30pIGRvZXMgbm90IG1hdGNoIHJhbmsgb2YgZHkgYCArXG4gICAgICAgICAgYCgkeyRkeS5yYW5rfSlgKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgICRkeS5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIG1heFBvb2xHcmFkOiBkeSBtdXN0IGJlIHJhbmsgNCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgYCR7JGR5LnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICRpbnB1dC5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIG1heFBvb2xHcmFkOiBpbnB1dCBtdXN0IGJlIHJhbmsgNCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgYCR7JGlucHV0LnJhbmt9LmApO1xuICBjb252X3V0aWwuY2hlY2tQYWRPbkRpbVJvdW5kaW5nTW9kZSgnbWF4UG9vbEdyYWQnLCBwYWQsIGRpbVJvdW5kaW5nTW9kZSk7XG4gIGNvbnN0IGlucHV0czogTWF4UG9vbEdyYWRJbnB1dHMgPSB7ZHk6ICRkeSwgaW5wdXQ6ICRpbnB1dCwgb3V0cHV0OiAkb3V0cHV0fTtcbiAgY29uc3QgYXR0cnM6IE1heFBvb2xHcmFkQXR0cnMgPSB7ZmlsdGVyU2l6ZSwgc3RyaWRlcywgcGFkLCBkaW1Sb3VuZGluZ01vZGV9O1xuXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgTWF4UG9vbEdyYWQsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKSBhcyBUZW5zb3I0RDtcbn1cblxuZXhwb3J0IGNvbnN0IG1heFBvb2xHcmFkID0gLyogQF9fUFVSRV9fICovIG9wKHttYXhQb29sR3JhZF99KTtcbiJdfQ==