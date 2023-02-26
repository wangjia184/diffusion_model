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
import { Conv2DBackpropFilter } from '../kernel_names';
import * as util from '../util';
import * as conv_util from './conv_util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the derivative of the filter of a 2D convolution.
 *
 * @param x The input tensor, of rank 4 or rank 3 of shape
 *     [batch, height, width, inChannels]. If rank 3, batch of 1 is assumed.
 * @param dy The dy image, of rank 4 or rank 3, of shape
 *     [batch, height, width, outDepth]. If rank 3, batch of 1 is assumed.
 * @param filterShape The shape of the filter, length 4,
 *     [filterHeight, filterWidth, inDepth, outDepth].
 * @param strides The strides of the convolution: [strideHeight,
 * strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 * @param dataFormat: An optional string from: "NHWC", "NCHW". Defaults to
 *     "NHWC". Specify the data format of the input and output data. With the
 *     default format "NHWC", the data is stored in the order of: [batch,
 *     height, width, channels].
 * @param dimRoundingMode A string from: 'ceil', 'round', 'floor'. If none is
 *     provided, it will default to truncate.
 */
function conv2DBackpropFilter_(x, dy, filterShape, strides, pad, dataFormat = 'NHWC', dimRoundingMode) {
    let x4D = x;
    if (x.rank === 3) {
        x4D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2]]);
    }
    let dy4D = dy;
    if (dy4D.rank === 3) {
        dy4D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2]]);
    }
    util.assert(x4D.rank === 4, () => `Error in conv2dDerFilter: input must be rank 4, but got shape ` +
        `${x4D.shape}.`);
    util.assert(dy4D.rank === 4, () => `Error in conv2dDerFilter: dy must be rank 4, but got shape ` +
        `${dy4D.shape}.`);
    util.assert(filterShape.length === 4, () => `Error in conv2dDerFilter: filterShape must be length 4, but got ` +
        `${filterShape}.`);
    const inDepth = dataFormat === 'NHWC' ? x4D.shape[3] : x4D.shape[1];
    const outDepth = dataFormat === 'NHWC' ? dy4D.shape[3] : dy4D.shape[1];
    util.assert(inDepth === filterShape[2], () => `Error in conv2dDerFilter: depth of input ${inDepth}) must ` +
        `match input depth in filter (${filterShape[2]}.`);
    util.assert(outDepth === filterShape[3], () => `Error in conv2dDerFilter: depth of dy (${outDepth}) must ` +
        `match output depth for filter (${filterShape[3]}).`);
    conv_util.checkPadOnDimRoundingMode('conv2dDerFilter', pad, dimRoundingMode);
    const inputs = { x: x4D, dy: dy4D };
    const attrs = { strides, pad, dataFormat, dimRoundingMode, filterShape };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    return ENGINE.runKernel(Conv2DBackpropFilter, inputs, attrs);
}
export const conv2DBackpropFilter = /* @__PURE__ */ op({ conv2DBackpropFilter_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjJkX2JhY2twcm9wX2ZpbHRlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2NvbnYyZF9iYWNrcHJvcF9maWx0ZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsb0JBQW9CLEVBQXdELE1BQU0saUJBQWlCLENBQUM7QUFJNUcsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxLQUFLLFNBQVMsTUFBTSxhQUFhLENBQUM7QUFDekMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBRWxDOzs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBbUJHO0FBQ0gsU0FBUyxxQkFBcUIsQ0FDMUIsQ0FBSSxFQUFFLEVBQUssRUFBRSxXQUE2QyxFQUMxRCxPQUFnQyxFQUNoQyxHQUFvRCxFQUNwRCxhQUE0QixNQUFNLEVBQ2xDLGVBQXdDO0lBQzFDLElBQUksR0FBRyxHQUFHLENBQWEsQ0FBQztJQUN4QixJQUFJLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ2hCLEdBQUcsR0FBRyxPQUFPLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUMzRDtJQUNELElBQUksSUFBSSxHQUFHLEVBQWMsQ0FBQztJQUMxQixJQUFJLElBQUksQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ25CLElBQUksR0FBRyxPQUFPLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUNoRTtJQUNELElBQUksQ0FBQyxNQUFNLENBQ1AsR0FBRyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2QsR0FBRyxFQUFFLENBQUMsZ0VBQWdFO1FBQ2xFLEdBQUcsR0FBRyxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUM7SUFDekIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDZixHQUFHLEVBQUUsQ0FBQyw2REFBNkQ7UUFDL0QsR0FBRyxJQUFJLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQztJQUMxQixJQUFJLENBQUMsTUFBTSxDQUNQLFdBQVcsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUN4QixHQUFHLEVBQUUsQ0FBQyxrRUFBa0U7UUFDcEUsR0FBRyxXQUFXLEdBQUcsQ0FBQyxDQUFDO0lBQzNCLE1BQU0sT0FBTyxHQUFHLFVBQVUsS0FBSyxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDcEUsTUFBTSxRQUFRLEdBQUcsVUFBVSxLQUFLLE1BQU0sQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN2RSxJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sS0FBSyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQzFCLEdBQUcsRUFBRSxDQUFDLDRDQUE0QyxPQUFPLFNBQVM7UUFDOUQsZ0NBQWdDLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDM0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLEtBQUssV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUMzQixHQUFHLEVBQUUsQ0FBQywwQ0FBMEMsUUFBUSxTQUFTO1FBQzdELGtDQUFrQyxXQUFXLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQzlELFNBQVMsQ0FBQyx5QkFBeUIsQ0FBQyxpQkFBaUIsRUFBRSxHQUFHLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDN0UsTUFBTSxNQUFNLEdBQStCLEVBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxFQUFFLEVBQUUsSUFBSSxFQUFDLENBQUM7SUFDOUQsTUFBTSxLQUFLLEdBQ1AsRUFBQyxPQUFPLEVBQUUsR0FBRyxFQUFFLFVBQVUsRUFBRSxlQUFlLEVBQUUsV0FBVyxFQUFDLENBQUM7SUFFN0QsMERBQTBEO0lBQzFELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDWixvQkFBb0IsRUFBRSxNQUFtQyxFQUN6RCxLQUFnQyxDQUFhLENBQUM7QUFDM0QsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLG9CQUFvQixHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxxQkFBcUIsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7Q29udjJEQmFja3Byb3BGaWx0ZXIsIENvbnYyREJhY2twcm9wRmlsdGVyQXR0cnMsIENvbnYyREJhY2twcm9wRmlsdGVySW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjNELCBUZW5zb3I0RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi9jb252X3V0aWwnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL3Jlc2hhcGUnO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBkZXJpdmF0aXZlIG9mIHRoZSBmaWx0ZXIgb2YgYSAyRCBjb252b2x1dGlvbi5cbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yLCBvZiByYW5rIDQgb3IgcmFuayAzIG9mIHNoYXBlXG4gKiAgICAgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBpbkNoYW5uZWxzXS4gSWYgcmFuayAzLCBiYXRjaCBvZiAxIGlzIGFzc3VtZWQuXG4gKiBAcGFyYW0gZHkgVGhlIGR5IGltYWdlLCBvZiByYW5rIDQgb3IgcmFuayAzLCBvZiBzaGFwZVxuICogICAgIFtiYXRjaCwgaGVpZ2h0LCB3aWR0aCwgb3V0RGVwdGhdLiBJZiByYW5rIDMsIGJhdGNoIG9mIDEgaXMgYXNzdW1lZC5cbiAqIEBwYXJhbSBmaWx0ZXJTaGFwZSBUaGUgc2hhcGUgb2YgdGhlIGZpbHRlciwgbGVuZ3RoIDQsXG4gKiAgICAgW2ZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGgsIGluRGVwdGgsIG91dERlcHRoXS5cbiAqIEBwYXJhbSBzdHJpZGVzIFRoZSBzdHJpZGVzIG9mIHRoZSBjb252b2x1dGlvbjogW3N0cmlkZUhlaWdodCxcbiAqIHN0cmlkZVdpZHRoXS5cbiAqIEBwYXJhbSBwYWQgQSBzdHJpbmcgZnJvbTogJ3NhbWUnLCAndmFsaWQnLiBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobVxuICogICAgIHVzZWQgaW4gdGhlIGZvcndhcmQgcHJvcCBvZiB0aGUgb3AuXG4gKiBAcGFyYW0gZGF0YUZvcm1hdDogQW4gb3B0aW9uYWwgc3RyaW5nIGZyb206IFwiTkhXQ1wiLCBcIk5DSFdcIi4gRGVmYXVsdHMgdG9cbiAqICAgICBcIk5IV0NcIi4gU3BlY2lmeSB0aGUgZGF0YSBmb3JtYXQgb2YgdGhlIGlucHV0IGFuZCBvdXRwdXQgZGF0YS4gV2l0aCB0aGVcbiAqICAgICBkZWZhdWx0IGZvcm1hdCBcIk5IV0NcIiwgdGhlIGRhdGEgaXMgc3RvcmVkIGluIHRoZSBvcmRlciBvZjogW2JhdGNoLFxuICogICAgIGhlaWdodCwgd2lkdGgsIGNoYW5uZWxzXS5cbiAqIEBwYXJhbSBkaW1Sb3VuZGluZ01vZGUgQSBzdHJpbmcgZnJvbTogJ2NlaWwnLCAncm91bmQnLCAnZmxvb3InLiBJZiBub25lIGlzXG4gKiAgICAgcHJvdmlkZWQsIGl0IHdpbGwgZGVmYXVsdCB0byB0cnVuY2F0ZS5cbiAqL1xuZnVuY3Rpb24gY29udjJEQmFja3Byb3BGaWx0ZXJfPFQgZXh0ZW5kcyBUZW5zb3IzRHxUZW5zb3I0RD4oXG4gICAgeDogVCwgZHk6IFQsIGZpbHRlclNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdfG51bWJlcixcbiAgICBwYWQ6ICd2YWxpZCd8J3NhbWUnfG51bWJlcnxjb252X3V0aWwuRXhwbGljaXRQYWRkaW5nLFxuICAgIGRhdGFGb3JtYXQ6ICdOSFdDJ3wnTkNIVycgPSAnTkhXQycsXG4gICAgZGltUm91bmRpbmdNb2RlPzogJ2Zsb29yJ3wncm91bmQnfCdjZWlsJyk6IFRlbnNvcjREIHtcbiAgbGV0IHg0RCA9IHggYXMgVGVuc29yNEQ7XG4gIGlmICh4LnJhbmsgPT09IDMpIHtcbiAgICB4NEQgPSByZXNoYXBlKHgsIFsxLCB4LnNoYXBlWzBdLCB4LnNoYXBlWzFdLCB4LnNoYXBlWzJdXSk7XG4gIH1cbiAgbGV0IGR5NEQgPSBkeSBhcyBUZW5zb3I0RDtcbiAgaWYgKGR5NEQucmFuayA9PT0gMykge1xuICAgIGR5NEQgPSByZXNoYXBlKGR5LCBbMSwgZHkuc2hhcGVbMF0sIGR5LnNoYXBlWzFdLCBkeS5zaGFwZVsyXV0pO1xuICB9XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgeDRELnJhbmsgPT09IDQsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY29udjJkRGVyRmlsdGVyOiBpbnB1dCBtdXN0IGJlIHJhbmsgNCwgYnV0IGdvdCBzaGFwZSBgICtcbiAgICAgICAgICBgJHt4NEQuc2hhcGV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGR5NEQucmFuayA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjb252MmREZXJGaWx0ZXI6IGR5IG11c3QgYmUgcmFuayA0LCBidXQgZ290IHNoYXBlIGAgK1xuICAgICAgICAgIGAke2R5NEQuc2hhcGV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGZpbHRlclNoYXBlLmxlbmd0aCA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjb252MmREZXJGaWx0ZXI6IGZpbHRlclNoYXBlIG11c3QgYmUgbGVuZ3RoIDQsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgYCR7ZmlsdGVyU2hhcGV9LmApO1xuICBjb25zdCBpbkRlcHRoID0gZGF0YUZvcm1hdCA9PT0gJ05IV0MnID8geDRELnNoYXBlWzNdIDogeDRELnNoYXBlWzFdO1xuICBjb25zdCBvdXREZXB0aCA9IGRhdGFGb3JtYXQgPT09ICdOSFdDJyA/IGR5NEQuc2hhcGVbM10gOiBkeTRELnNoYXBlWzFdO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGluRGVwdGggPT09IGZpbHRlclNoYXBlWzJdLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGNvbnYyZERlckZpbHRlcjogZGVwdGggb2YgaW5wdXQgJHtpbkRlcHRofSkgbXVzdCBgICtcbiAgICAgICAgICBgbWF0Y2ggaW5wdXQgZGVwdGggaW4gZmlsdGVyICgke2ZpbHRlclNoYXBlWzJdfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBvdXREZXB0aCA9PT0gZmlsdGVyU2hhcGVbM10sXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY29udjJkRGVyRmlsdGVyOiBkZXB0aCBvZiBkeSAoJHtvdXREZXB0aH0pIG11c3QgYCArXG4gICAgICAgICAgYG1hdGNoIG91dHB1dCBkZXB0aCBmb3IgZmlsdGVyICgke2ZpbHRlclNoYXBlWzNdfSkuYCk7XG4gIGNvbnZfdXRpbC5jaGVja1BhZE9uRGltUm91bmRpbmdNb2RlKCdjb252MmREZXJGaWx0ZXInLCBwYWQsIGRpbVJvdW5kaW5nTW9kZSk7XG4gIGNvbnN0IGlucHV0czogQ29udjJEQmFja3Byb3BGaWx0ZXJJbnB1dHMgPSB7eDogeDRELCBkeTogZHk0RH07XG4gIGNvbnN0IGF0dHJzOiBDb252MkRCYWNrcHJvcEZpbHRlckF0dHJzID1cbiAgICAgIHtzdHJpZGVzLCBwYWQsIGRhdGFGb3JtYXQsIGRpbVJvdW5kaW5nTW9kZSwgZmlsdGVyU2hhcGV9O1xuXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgcmV0dXJuIEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgQ29udjJEQmFja3Byb3BGaWx0ZXIsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKSBhcyBUZW5zb3I0RDtcbn1cblxuZXhwb3J0IGNvbnN0IGNvbnYyREJhY2twcm9wRmlsdGVyID0gLyogQF9fUFVSRV9fICovIG9wKHtjb252MkRCYWNrcHJvcEZpbHRlcl99KTtcbiJdfQ==