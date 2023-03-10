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
import { Conv3DBackpropFilterV2 } from '../kernel_names';
import * as util from '../util';
import { op } from './operation';
import { reshape } from './reshape';
/**
 * Computes the derivative of the filter of a 3D convolution.
 *
 * @param x The input tensor, of rank 5 or rank 4 of shape
 *     [batch, depth, height, width, inChannels]. If rank 4, batch of 1 is
 *     assumed.
 * @param dy The dy image, of rank 5 or rank 4, of shape
 *     [batch, depth, height, width, outDepth]. If rank 4, batch of 1 is
 *     assumed.
 * @param filterShape The shape of the filter, length 5,
 *     [filterDepth, filterHeight, filterWidth, inDepth, outDepth].
 * @param strides The strides of the convolution: [strideDepth, strideHeight,
 * strideWidth].
 * @param pad A string from: 'same', 'valid'. The type of padding algorithm
 *     used in the forward prop of the op.
 */
function conv3DBackpropFilter_(x, dy, filterShape, strides, pad) {
    let x5D = x;
    if (x.rank === 4) {
        x5D = reshape(x, [1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]]);
    }
    let dy5D = dy;
    if (dy5D.rank === 4) {
        dy5D = reshape(dy, [1, dy.shape[0], dy.shape[1], dy.shape[2], dy.shape[3]]);
    }
    util.assert(x5D.rank === 5, () => `Error in conv3dDerFilter: input must be rank 5, but got shape ` +
        `${x5D.shape}.`);
    util.assert(dy5D.rank === 5, () => `Error in conv3dDerFilter: dy must be rank 5, but got shape ` +
        `${dy5D.shape}.`);
    util.assert(filterShape.length === 5, () => `Error in conv3dDerFilter: filterShape must be length 5, but got ` +
        `${filterShape}.`);
    util.assert(x5D.shape[4] === filterShape[3], () => `Error in conv3dDerFilter: depth of input ${x5D.shape[4]}) must ` +
        `match input depth in filter (${filterShape[3]}.`);
    util.assert(dy5D.shape[4] === filterShape[4], () => `Error in conv3dDerFilter: depth of dy (${dy5D.shape[4]}) must ` +
        `match output depth for filter (${filterShape[4]}).`);
    const inputs = { x: x5D, dy: dy5D };
    const attrs = { strides, pad, filterShape };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    return ENGINE.runKernel(Conv3DBackpropFilterV2, inputs, attrs);
}
export const conv3DBackpropFilter = /* @__PURE__ */ op({ conv3DBackpropFilter_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udjNkX2JhY2twcm9wX2ZpbHRlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL2NvbnYzZF9iYWNrcHJvcF9maWx0ZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsc0JBQXNCLEVBQTRELE1BQU0saUJBQWlCLENBQUM7QUFJbEgsT0FBTyxLQUFLLElBQUksTUFBTSxTQUFTLENBQUM7QUFFaEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBRWxDOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUNILFNBQVMscUJBQXFCLENBQzFCLENBQUksRUFBRSxFQUFLLEVBQUUsV0FBcUQsRUFDbEUsT0FBd0MsRUFBRSxHQUFtQjtJQUMvRCxJQUFJLEdBQUcsR0FBRyxDQUFhLENBQUM7SUFDeEIsSUFBSSxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNoQixHQUFHLEdBQUcsT0FBTyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUN2RTtJQUNELElBQUksSUFBSSxHQUFHLEVBQWMsQ0FBQztJQUMxQixJQUFJLElBQUksQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ25CLElBQUksR0FBRyxPQUFPLENBQUMsRUFBRSxFQUFFLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQzdFO0lBQ0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxHQUFHLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDZCxHQUFHLEVBQUUsQ0FBQyxnRUFBZ0U7UUFDbEUsR0FBRyxHQUFHLENBQUMsS0FBSyxHQUFHLENBQUMsQ0FBQztJQUN6QixJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNmLEdBQUcsRUFBRSxDQUFDLDZEQUE2RDtRQUMvRCxHQUFHLElBQUksQ0FBQyxLQUFLLEdBQUcsQ0FBQyxDQUFDO0lBQzFCLElBQUksQ0FBQyxNQUFNLENBQ1AsV0FBVyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQ3hCLEdBQUcsRUFBRSxDQUFDLGtFQUFrRTtRQUNwRSxHQUFHLFdBQVcsR0FBRyxDQUFDLENBQUM7SUFDM0IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFDL0IsR0FBRyxFQUFFLENBQUMsNENBQTRDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLFNBQVM7UUFDbkUsZ0NBQWdDLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDM0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFDaEMsR0FBRyxFQUFFLENBQUMsMENBQTBDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLFNBQVM7UUFDbEUsa0NBQWtDLFdBQVcsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7SUFFOUQsTUFBTSxNQUFNLEdBQWlDLEVBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxFQUFFLEVBQUUsSUFBSSxFQUFDLENBQUM7SUFFaEUsTUFBTSxLQUFLLEdBQWdDLEVBQUMsT0FBTyxFQUFFLEdBQUcsRUFBRSxXQUFXLEVBQUMsQ0FBQztJQUV2RSwwREFBMEQ7SUFDMUQsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNaLHNCQUFzQixFQUFFLE1BQW1DLEVBQzNELEtBQWdDLENBQWEsQ0FBQztBQUMzRCxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sb0JBQW9CLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLHFCQUFxQixFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtDb252M0RCYWNrcHJvcEZpbHRlclYyLCBDb252M0RCYWNrcHJvcEZpbHRlclYyQXR0cnMsIENvbnYzREJhY2twcm9wRmlsdGVyVjJJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yNEQsIFRlbnNvcjVEfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi9yZXNoYXBlJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgZGVyaXZhdGl2ZSBvZiB0aGUgZmlsdGVyIG9mIGEgM0QgY29udm9sdXRpb24uXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IHRlbnNvciwgb2YgcmFuayA1IG9yIHJhbmsgNCBvZiBzaGFwZVxuICogICAgIFtiYXRjaCwgZGVwdGgsIGhlaWdodCwgd2lkdGgsIGluQ2hhbm5lbHNdLiBJZiByYW5rIDQsIGJhdGNoIG9mIDEgaXNcbiAqICAgICBhc3N1bWVkLlxuICogQHBhcmFtIGR5IFRoZSBkeSBpbWFnZSwgb2YgcmFuayA1IG9yIHJhbmsgNCwgb2Ygc2hhcGVcbiAqICAgICBbYmF0Y2gsIGRlcHRoLCBoZWlnaHQsIHdpZHRoLCBvdXREZXB0aF0uIElmIHJhbmsgNCwgYmF0Y2ggb2YgMSBpc1xuICogICAgIGFzc3VtZWQuXG4gKiBAcGFyYW0gZmlsdGVyU2hhcGUgVGhlIHNoYXBlIG9mIHRoZSBmaWx0ZXIsIGxlbmd0aCA1LFxuICogICAgIFtmaWx0ZXJEZXB0aCwgZmlsdGVySGVpZ2h0LCBmaWx0ZXJXaWR0aCwgaW5EZXB0aCwgb3V0RGVwdGhdLlxuICogQHBhcmFtIHN0cmlkZXMgVGhlIHN0cmlkZXMgb2YgdGhlIGNvbnZvbHV0aW9uOiBbc3RyaWRlRGVwdGgsIHN0cmlkZUhlaWdodCxcbiAqIHN0cmlkZVdpZHRoXS5cbiAqIEBwYXJhbSBwYWQgQSBzdHJpbmcgZnJvbTogJ3NhbWUnLCAndmFsaWQnLiBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobVxuICogICAgIHVzZWQgaW4gdGhlIGZvcndhcmQgcHJvcCBvZiB0aGUgb3AuXG4gKi9cbmZ1bmN0aW9uIGNvbnYzREJhY2twcm9wRmlsdGVyXzxUIGV4dGVuZHMgVGVuc29yNER8VGVuc29yNUQ+KFxuICAgIHg6IFQsIGR5OiBULCBmaWx0ZXJTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl18bnVtYmVyLCBwYWQ6ICd2YWxpZCd8J3NhbWUnKTogVGVuc29yNUQge1xuICBsZXQgeDVEID0geCBhcyBUZW5zb3I1RDtcbiAgaWYgKHgucmFuayA9PT0gNCkge1xuICAgIHg1RCA9IHJlc2hhcGUoeCwgWzEsIHguc2hhcGVbMF0sIHguc2hhcGVbMV0sIHguc2hhcGVbMl0sIHguc2hhcGVbM11dKTtcbiAgfVxuICBsZXQgZHk1RCA9IGR5IGFzIFRlbnNvcjVEO1xuICBpZiAoZHk1RC5yYW5rID09PSA0KSB7XG4gICAgZHk1RCA9IHJlc2hhcGUoZHksIFsxLCBkeS5zaGFwZVswXSwgZHkuc2hhcGVbMV0sIGR5LnNoYXBlWzJdLCBkeS5zaGFwZVszXV0pO1xuICB9XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgeDVELnJhbmsgPT09IDUsXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY29udjNkRGVyRmlsdGVyOiBpbnB1dCBtdXN0IGJlIHJhbmsgNSwgYnV0IGdvdCBzaGFwZSBgICtcbiAgICAgICAgICBgJHt4NUQuc2hhcGV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGR5NUQucmFuayA9PT0gNSxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjb252M2REZXJGaWx0ZXI6IGR5IG11c3QgYmUgcmFuayA1LCBidXQgZ290IHNoYXBlIGAgK1xuICAgICAgICAgIGAke2R5NUQuc2hhcGV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIGZpbHRlclNoYXBlLmxlbmd0aCA9PT0gNSxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjb252M2REZXJGaWx0ZXI6IGZpbHRlclNoYXBlIG11c3QgYmUgbGVuZ3RoIDUsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgYCR7ZmlsdGVyU2hhcGV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHg1RC5zaGFwZVs0XSA9PT0gZmlsdGVyU2hhcGVbM10sXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY29udjNkRGVyRmlsdGVyOiBkZXB0aCBvZiBpbnB1dCAke3g1RC5zaGFwZVs0XX0pIG11c3QgYCArXG4gICAgICAgICAgYG1hdGNoIGlucHV0IGRlcHRoIGluIGZpbHRlciAoJHtmaWx0ZXJTaGFwZVszXX0uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgZHk1RC5zaGFwZVs0XSA9PT0gZmlsdGVyU2hhcGVbNF0sXG4gICAgICAoKSA9PiBgRXJyb3IgaW4gY29udjNkRGVyRmlsdGVyOiBkZXB0aCBvZiBkeSAoJHtkeTVELnNoYXBlWzRdfSkgbXVzdCBgICtcbiAgICAgICAgICBgbWF0Y2ggb3V0cHV0IGRlcHRoIGZvciBmaWx0ZXIgKCR7ZmlsdGVyU2hhcGVbNF19KS5gKTtcblxuICBjb25zdCBpbnB1dHM6IENvbnYzREJhY2twcm9wRmlsdGVyVjJJbnB1dHMgPSB7eDogeDVELCBkeTogZHk1RH07XG5cbiAgY29uc3QgYXR0cnM6IENvbnYzREJhY2twcm9wRmlsdGVyVjJBdHRycyA9IHtzdHJpZGVzLCBwYWQsIGZpbHRlclNoYXBlfTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgICAgICAgIENvbnYzREJhY2twcm9wRmlsdGVyVjIsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKSBhcyBUZW5zb3I1RDtcbn1cblxuZXhwb3J0IGNvbnN0IGNvbnYzREJhY2twcm9wRmlsdGVyID0gLyogQF9fUFVSRV9fICovIG9wKHtjb252M0RCYWNrcHJvcEZpbHRlcl99KTtcbiJdfQ==