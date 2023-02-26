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
import { Conv2DBackpropInput } from '../kernel_names';
import * as util from '../util';
import * as conv_util from './conv_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the derivative of the input of a 2D convolution.
 *
 * @param xShape The shape of the input: [batch, height, width, inDepth].
 * If length of 3, batch of 1 is assumed.
 * @param dy The derivative of the output, of rank 4 or rank 3 of shape
 *   `[batch, outHeight, outWidth, outDepth]`. If rank 3, batch of 1 is
 * assumed.
 * @param filter The filter, rank 4, of shape
 *     `[filterHeight, filterWidth, inDepth, outDepth]`.
 * @param strides The strides of the convolution: `[strideHeight,
 * strideWidth]`.
 * @param pad The type of padding algorithm used:
 *    - `same` and stride 1: output will be of same size as input,
 *       regardless of filter size.
 *    - `valid`: output will be smaller than input if filter is larger
 *       than 1x1.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function conv2DBackpropInput_(xShape, dy, filter, strides, pad, dataFormat = 'NHWC', dimRoundingMode) {
    util.assert(xShape.length === dy.rank, () => `Length of inShape ` +
        `(${xShape.length}) and rank of dy (${dy.rank}) must match`);
    let xShape4D = xShape;
    let dy4D = dy;
    let reshapedTo4D = false;
    if (dy.rank === 3) {
        reshapedTo4D = true;
        dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
        xShape4D = [1, xShape[0], xShape[1], xShape[2]];
    }
    util.assert(xShape4D.length === 4, () => `Error in conv2dDerInput: inShape must be length 4, but got length ` +
        `${xShape4D.length}.`);
    util.assert(dy4D.rank === 4, () => `Error in conv2dDerInput: dy must be rank 4, but got ` +
        `rank ${dy4D.rank}`);
    util.assert(filter.rank === 4, () => `Error in conv2dDerInput: filter must be rank 4, but got ` +
        `rank ${filter.rank}`);
    const inDepth = dataFormat === 'NHWC' ? xShape4D[3] : xShape4D[1];
    const outDepth = dataFormat === 'NHWC' ? dy4D.shape[3] : dy4D.shape[1];
    util.assert(inDepth === filter.shape[2], () => `Error in conv2dDerInput: depth of input (${inDepth}) must ` +
        `match input depth for filter ${filter.shape[2]}.`);
    util.assert(outDepth === filter.shape[3], () => `Error in conv2dDerInput: depth of output (${outDepth}) must ` +
        `match output depth for filter ${filter.shape[3]}.`);
    conv_util.checkPadOnDimRoundingMode('conv2dDerInput', pad, dimRoundingMode);
    const inputs = { dy: dy4D, filter };
    const attrs = { strides, pad, dataFormat, dimRoundingMode, inputShape: xShape4D };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(Conv2DBackpropInput, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
export const conv2DBackpropInput = /* @__PURE__ */ op({ conv2DBackpropInput_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjJkX2JhY2twcm9wX2lucHV0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvY29udjJkX2JhY2twcm9wX2lucHV0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLG1CQUFtQixFQUFzRCxNQUFNLGlCQUFpQixDQUFDO0FBSXpHLE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sS0FBSyxTQUFTLE1BQU0sYUFBYSxDQUFDO0FBQ3pDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUVsQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0F1Qkc7QUFDSCxTQUFTLG9CQUFvQixDQUN6QixNQUFpRSxFQUFFLEVBQUssRUFDeEUsTUFBZ0IsRUFBRSxPQUFnQyxFQUNsRCxHQUFvRCxFQUNwRCxhQUE0QixNQUFNLEVBQ2xDLGVBQXdDO0lBQzFDLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLE1BQU0sS0FBSyxFQUFFLENBQUMsSUFBSSxFQUN6QixHQUFHLEVBQUUsQ0FBQyxvQkFBb0I7UUFDdEIsSUFBSSxNQUFNLENBQUMsTUFBTSxxQkFBcUIsRUFBRSxDQUFDLElBQUksY0FBYyxDQUFDLENBQUM7SUFFckUsSUFBSSxRQUFRLEdBQUcsTUFBMEMsQ0FBQztJQUMxRCxJQUFJLElBQUksR0FBRyxFQUFjLENBQUM7SUFDMUIsSUFBSSxZQUFZLEdBQUcsS0FBSyxDQUFDO0lBQ3pCLElBQUksRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDakIsWUFBWSxHQUFHLElBQUksQ0FBQztRQUNwQixJQUFJLEdBQUcsT0FBTyxDQUFD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Ii8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtDb252MkRCYWNrcHJvcElucHV0LCBDb252MkRCYWNrcHJvcElucHV0QXR0cnMsIENvbnYyREJhY2twcm9wSW5wdXRJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yM0QsIFRlbnNvcjREfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuL2NvbnZfdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5cbi8qKlxuICogQ29tcHV0ZXMgdGhlIGRlcml2YXRpdmUgb2YgdGhlIGlucHV0IG9mIGEgMkQgY29udm9sdXRpb24uXG4gKlxuICogQHBhcmFtIHhTaGFwZSBUaGUgc2hhcGUgb2YgdGhlIGlucHV0OiBbYmF0Y2gsIGhlaWdodCwgd2lkdGgsIGluRGVwdGhdLlxuICogSWYgbGVuZ3RoIG9mIDMsIGJhdGNoIG9mIDEgaXMgYXNzdW1lZC5cbiAqIEBwYXJhbSBkeSBUaGUgZGVyaXZhdGl2ZSBvZiB0aGUgb3V0cHV0LCBvZiByYW5rIDQgb3IgcmFuayAzIG9mIHNoYXBlXG4gKiAgIGBbYmF0Y2gsIG91dEhlaWdodCwgb3V0V2lkdGgsIG91dERlcHRoXWAuIElmIHJhbmsgMywgYmF0Y2ggb2YgMSBpc1xuICogYXNzdW1lZC5cbiAqIEBwYXJhbSBmaWx0ZXIgVGhlIGZpbHRlciwgcmFuayA0LCBvZiBzaGFwZVxuICogICAgIGBbZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgaW5EZXB0aCwgb3V0RGVwdGhdYC5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBjb252b2x1dGlvbjogYFtzdHJpZGVIZWlnaHQsXG4gKiBzdHJpZGVXaWR0aF1gLlxuICogQHBhcmFtIHBhZCBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobSB1c2VkOlxuICogICAgLSBgc2FtZWAgYW5kIHN0cmlkZSAxOiBvdXRwdXQgd2lsbCBiZSBvZiBzYW1lIHNpemUgYXMgaW5wdXQsXG4gKiAgICAgICByZWdhcmRsZXNzIG9mIGZpbHRlciBzaXplLlxuICogICAgLSBgdmFsaWRgOiBvdXRwdXQgd2lsbCBiZSBzbWFsbGVyIHRoYW4gaW5wdXQgaWYgZmlsdGVyIGlzIGxhcmdlclxuICogICAgICAgdGhhbiAxeDEuXG4gKiBAcGFyYW0gZGF0YUZvcm1hdDogQW4gb3B0aW9uYWwgc3RyaW5nIGZyb206IFwiTkhXQ1wiLCBcIk5DSFdcIi4gRGVmYXVsdHMgdG9cbiAqICAgICBcIk5IV0NcIi4gU3BlY2lmeSB0aGUgZGF0YSBmb3JtYXQgb2YgdGhlIGlucHV0IGFuZCBvdXRwdXQgZGF0YS4gV2l0aCB0aGVcbiAqICAgICBkZWZhdWx0IGZvcm1hdCBcIk5IV0NcIiwgdGhlIGRhdGEgaXMgc3RvcmVkIGluIHRoZSBvcmRlciBvZjogW2JhdGNoLFxuICogICAgIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzXS5cbiAqIEBwYXJhbSBkaW1Sb3VuZGluZ01vZGUgQSBzdHJpbmcgZnJvbTogJ2NlaWwnLCAncm91bmQnLCAnZmxvb3InLiBJZiBub25lIGlzXG4gKiAgICAgcHJvdmlkZWQsIGl0IHdpbGwgZGVmYXVsdCB0byB0cnVuY2F0ZS5cbiAqL1xuZnVuY3Rpb24gY29udjJEQmFja3Byb3BJbnB1dF88VCBleHRlbmRzIFRlbnNvcjNEfFRlbnNvcjREPihcbiAgICB4U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZHk6IFQsXG4gICAgZmlsdGVyOiBUZW5zb3I0RCwgc3RyaWRlczogW251bWJlciwgbnVtYmVyXXxudW1iZXIsXG4gICAgcGFkOiAndmFsaWQnfCdzYW1lJ3xudW1iZXJ8Y29udl91dGlsLkV4cGxpY2l0UGFkZGluZyxcbiAgICBkYXRhRm9ybWF0OiAnTkhXQyd8J05DSFcnID0gJ05IV0MnLFxuICAgIGRpbVJvdW5kaW5nTW9kZT86ICdmbG9vcid8J3JvdW5kJ3wnY2VpbCcpOiBUIHtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4U2hhcGUubGVuZ3RoID09PSBkeS5yYW5rLFxuICAgICAgKCkgPT4gYExlbmd0aCBvZiBpblNoYXBlIGAgK1xuICAgICAgICAgIGAoJHt4U2hhcGUubGVuZ3RofSkgYW5kIHJhbmsgb2YgZHkgKCR7ZHkucmFua30pIG11c3QgbWF0Y2hgKTtcblxuICBsZXQgeFNoYXBlNEQgPSB4U2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIGxldCBkeTREID0gZHkgYXMgVGVuc29yNEQ7XG4gIGxldCByZXNoYXBlZFRvNEQgPSBmYWxzZTtcbiAgaWYgKGR5LnJhbmsgPT09IDMpIHtcbiAgICByZXNoYXBlZFRvNEQgPSB0cnVlO1xuICAgIGR5NEQgPSByZXNoYXBlKGR5LCBbMSwgZHkuc2hhcGVbMF0sIGR5LnNoYXBlWzFdLCBkeS5zaGFwZVsyXV0pO1xuICAgIHhTaGFwZTREID0gWzEsIHhTaGFwZVswXSwgeFNoYXBlWzFdLCB4U2hhcGVbMl1dO1xuICB9XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4U2hhcGU0RC5sZW5ndGggPT09IDQsXG4gICAgICAoKSA9PlxuICAgICAgICAgIGBFcnJvciBpbiBjb252MmREZXJJbnB1dDogaW5TaGFwZSBtdXN0IGJlIGxlbmd0aCA0LCBidXQgZ290IGxlbmd0aCBgICtcbiAgICAgICAgICBgJHt4U2hhcGU0RC5sZW5ndGh9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGR5NEQucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjb252MmREZXJJbnB1dDogZHkgbXVzdCBiZSByYW5rIDQsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgYHJhbmsgJHtkeTRELnJhbmt9YCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgZmlsdGVyLnJhbmsgPT09IDQsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY29udjJkRGVySW5wdXQ6IGZpbHRlciBtdXN0IGJlIHJhbmsgNCwgYnV0IGdvdCBgICtcbiAgICAgICAgICBgcmFuayAke2ZpbHRlci5yYW5rfWApO1xuICBjb25zdCBpbkRlcHRoID0gZGF0YUZvcm1hdCA9PT0gJ05IV0MnID8geFNoYXBlNERbM10gOiB4U2hhcGU0RFsxXTtcbiAgY29uc3Qgb3V0RGVwdGggPSBkYXRhRm9ybWF0ID09PSAnTkhXQycgPyBkeTRELnNoYXBlWzNdIDogZHk0RC5zaGFwZVsxXTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBpbkRlcHRoID09PSBmaWx0ZXIuc2hhcGVbMl0sXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY29udjJkRGVySW5wdXQ6IGRlcHRoIG9mIGlucHV0ICgke2luRGVwdGh9KSBtdXN0IGAgK1xuICAgICAgICAgIGBtYXRjaCBpbnB1dCBkZXB0aCBmb3IgZmlsdGVyICR7ZmlsdGVyLnNoYXBlWzJdfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBvdXREZXB0aCA9PT0gZmlsdGVyLnNoYXBlWzNdLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGNvbnYyZERlcklucHV0OiBkZXB0aCBvZiBvdXRwdXQgKCR7b3V0RGVwdGh9KSBtdXN0IGAgK1xuICAgICAgICAgIGBtYXRjaCBvdXRwdXQgZGVwdGggZm9yIGZpbHRlciAke2ZpbHRlci5zaGFwZVszXX0uYCk7XG4gIGNvbnZfdXRpbC5jaGVja1BhZE9uRGltUm91bmRpbmdNb2RlKCdjb252MmREZXJJbnB1dCcsIHBhZCwgZGltUm91bmRpbmdNb2RlKTtcbiAgY29uc3QgaW5wdXRzOiBDb252MkRCYWNrcHJvcElucHV0SW5wdXRzID0ge2R5OiBkeTRELCBmaWx0ZXJ9O1xuICBjb25zdCBhdHRyczogQ29udjJEQmFja3Byb3BJbnB1dEF0dHJzID1cbiAgICAgIHtzdHJpZGVzLCBwYWQsIGRhdGFGb3JtYXQsIGRpbVJvdW5kaW5nTW9kZSwgaW5wdXRTaGFwZTogeFNoYXBlNER9O1xuXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgY29uc3QgcmVzID0gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICAgICAgICAgIENvbnYyREJhY2twcm9wSW5wdXQsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFQ7XG5cbiAgaWYgKHJlc2hhcGVkVG80RCkge1xuICAgIHJldHVybiByZXNoYXBlKHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM11dKSBhcyBUO1xuICB9XG4gIHJldHVybiByZXM7XG59XG5cbmV4cG9ydCBjb25zdCBjb252MkRCYWNrcHJvcElucHV0ID0gLyogQF9fUFVSRV9fICovIG9wKHtjb252MkRCYWNrcHJvcElucHV0X30pO1xuIl19