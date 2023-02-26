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
import { DenseBincount } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
/**
 * Outputs a vector with length `size` and the same dtype as `weights`.
 *
 * If `weights` are empty, then index `i` stores the number of times the value
 * `i` is counted in `x`. If `weights` are non-empty, then index `i` stores the
 * sum of the value in `weights` at each index where the corresponding value in
 * `x` is `i`.
 *
 * Values in `x` outside of the range [0, size) are ignored.
 *
 * @param x The input int tensor, rank 1 or rank 2.
 * @param weights The weights tensor, must have the same shape as x, or a
 *     length-0 Tensor, in which case it acts as all weights equal to 1.
 * @param size Non-negative integer.
 * @param binaryOutput Optional. Whether the kernel should count the appearance
 *     or number of occurrences. Defaults to False.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function denseBincount_(x, weights, size, binaryOutput = false) {
    const $x = convertToTensor(x, 'x', 'denseBincount');
    const $weights = convertToTensor(weights, 'weights', 'denseBincount');
    util.assert($x.dtype === 'int32', () => `Error in denseBincount: input ` +
        `dtype must be int32, but got ${$x.dtype}`);
    util.assert($x.rank <= 2, () => `Error in denseBincount: input must be at most rank 2, but got ` +
        `rank ${$x.rank}.`);
    util.assert(size >= 0, () => `size must be non-negative, but got ${size}.`);
    util.assert($weights.size === $x.size || $weights.size === 0, () => `Error in denseBincount: weights must have the same shape as x or ` +
        `0-length, but got x shape: ${$x.shape}, weights shape: ` +
        `${$weights.shape}.`);
    const inputs = { x: $x, weights: $weights };
    const attrs = { size, binaryOutput };
    return ENGINE.runKernel(DenseBincount, inputs, attrs);
}
export const denseBincount = /* @__PURE__ */ op({ denseBincount_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGVuc2VfYmluY291bnQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9kZW5zZV9iaW5jb3VudC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxhQUFhLEVBQTBDLE1BQU0saUJBQWlCLENBQUM7QUFJdkYsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQWtCRztBQUNILFNBQVMsY0FBYyxDQUNuQixDQUFlLEVBQUUsT0FBcUIsRUFBRSxJQUFZLEVBQ3BELFlBQVksR0FBRyxLQUFLO0lBQ3RCLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLGVBQWUsQ0FBQyxDQUFDO0lBQ3BELE1BQU0sUUFBUSxHQUFHLGVBQWUsQ0FBQyxPQUFPLEVBQUUsU0FBUyxFQUFFLGVBQWUsQ0FBQyxDQUFDO0lBRXRFLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLEtBQUssS0FBSyxPQUFPLEVBQ3BCLEdBQUcsRUFBRSxDQUFDLGdDQUFnQztRQUNsQyxnQ0FBZ0MsRUFBRSxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUM7SUFDcEQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxJQUFJLENBQUMsRUFDWixHQUFHLEVBQUUsQ0FBQyxnRUFBZ0U7UUFDbEUsUUFBUSxFQUFFLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUM1QixJQUFJLENBQUMsTUFBTSxDQUFDLElBQUksSUFBSSxDQUFDLEVBQUUsR0FBRyxFQUFFLENBQUMsc0NBQXNDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDNUUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsSUFBSSxLQUFLLEVBQUUsQ0FBQyxJQUFJLElBQUksUUFBUSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2hELEdBQUcsRUFBRSxDQUNELG1FQUFtRTtRQUNuRSw4QkFBOEIsRUFBRSxDQUFDLEtBQUssbUJBQW1CO1FBQ3pELEdBQUcsUUFBUSxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUM7SUFFOUIsTUFBTSxNQUFNLEdBQXdCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxPQUFPLEVBQUUsUUFBUSxFQUFDLENBQUM7SUFDL0QsTUFBTSxLQUFLLEdBQXVCLEVBQUMsSUFBSSxFQUFFLFlBQVksRUFBQyxDQUFDO0lBRXZELE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsYUFBYSxFQUFFLE1BQW1DLEVBQ2xELEtBQWdDLENBQUMsQ0FBQztBQUN4QyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sYUFBYSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxjQUFjLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7RGVuc2VCaW5jb3VudCwgRGVuc2VCaW5jb3VudEF0dHJzLCBEZW5zZUJpbmNvdW50SW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjFELCBUZW5zb3IyRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIE91dHB1dHMgYSB2ZWN0b3Igd2l0aCBsZW5ndGggYHNpemVgIGFuZCB0aGUgc2FtZSBkdHlwZSBhcyBgd2VpZ2h0c2AuXG4gKlxuICogSWYgYHdlaWdodHNgIGFyZSBlbXB0eSwgdGhlbiBpbmRleCBgaWAgc3RvcmVzIHRoZSBudW1iZXIgb2YgdGltZXMgdGhlIHZhbHVlXG4gKiBgaWAgaXMgY291bnRlZCBpbiBgeGAuIElmIGB3ZWlnaHRzYCBhcmUgbm9uLWVtcHR5LCB0aGVuIGluZGV4IGBpYCBzdG9yZXMgdGhlXG4gKiBzdW0gb2YgdGhlIHZhbHVlIGluIGB3ZWlnaHRzYCBhdCBlYWNoIGluZGV4IHdoZXJlIHRoZSBjb3JyZXNwb25kaW5nIHZhbHVlIGluXG4gKiBgeGAgaXMgYGlgLlxuICpcbiAqIFZhbHVlcyBpbiBgeGAgb3V0c2lkZSBvZiB0aGUgcmFuZ2UgWzAsIHNpemUpIGFyZSBpZ25vcmVkLlxuICpcbiAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbnQgdGVuc29yLCByYW5rIDEgb3IgcmFuayAyLlxuICogQHBhcmFtIHdlaWdodHMgVGhlIHdlaWdodHMgdGVuc29yLCBtdXN0IGhhdmUgdGhlIHNhbWUgc2hhcGUgYXMgeCwgb3IgYVxuICogICAgIGxlbmd0aC0wIFRlbnNvciwgaW4gd2hpY2ggY2FzZSBpdCBhY3RzIGFzIGFsbCB3ZWlnaHRzIGVxdWFsIHRvIDEuXG4gKiBAcGFyYW0gc2l6ZSBOb24tbmVnYXRpdmUgaW50ZWdlci5cbiAqIEBwYXJhbSBiaW5hcnlPdXRwdXQgT3B0aW9uYWwuIFdoZXRoZXIgdGhlIGtlcm5lbCBzaG91bGQgY291bnQgdGhlIGFwcGVhcmFuY2VcbiAqICAgICBvciBudW1iZXIgb2Ygb2NjdXJyZW5jZXMuIERlZmF1bHRzIHRvIEZhbHNlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ1JlZHVjdGlvbid9XG4gKi9cbmZ1bmN0aW9uIGRlbnNlQmluY291bnRfPFQgZXh0ZW5kcyBUZW5zb3IxRHxUZW5zb3IyRD4oXG4gICAgeDogVHxUZW5zb3JMaWtlLCB3ZWlnaHRzOiBUfFRlbnNvckxpa2UsIHNpemU6IG51bWJlcixcbiAgICBiaW5hcnlPdXRwdXQgPSBmYWxzZSk6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdkZW5zZUJpbmNvdW50Jyk7XG4gIGNvbnN0ICR3ZWlnaHRzID0gY29udmVydFRvVGVuc29yKHdlaWdodHMsICd3ZWlnaHRzJywgJ2RlbnNlQmluY291bnQnKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgICR4LmR0eXBlID09PSAnaW50MzInLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGRlbnNlQmluY291bnQ6IGlucHV0IGAgK1xuICAgICAgICAgIGBkdHlwZSBtdXN0IGJlIGludDMyLCBidXQgZ290ICR7JHguZHR5cGV9YCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJHgucmFuayA8PSAyLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGRlbnNlQmluY291bnQ6IGlucHV0IG11c3QgYmUgYXQgbW9zdCByYW5rIDIsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgYHJhbmsgJHskeC5yYW5rfS5gKTtcbiAgdXRpbC5hc3NlcnQoc2l6ZSA+PSAwLCAoKSA9PiBgc2l6ZSBtdXN0IGJlIG5vbi1uZWdhdGl2ZSwgYnV0IGdvdCAke3NpemV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICR3ZWlnaHRzLnNpemUgPT09ICR4LnNpemUgfHwgJHdlaWdodHMuc2l6ZSA9PT0gMCxcbiAgICAgICgpID0+XG4gICAgICAgICAgYEVycm9yIGluIGRlbnNlQmluY291bnQ6IHdlaWdodHMgbXVzdCBoYXZlIHRoZSBzYW1lIHNoYXBlIGFzIHggb3IgYCArXG4gICAgICAgICAgYDAtbGVuZ3RoLCBidXQgZ290IHggc2hhcGU6ICR7JHguc2hhcGV9LCB3ZWlnaHRzIHNoYXBlOiBgICtcbiAgICAgICAgICBgJHskd2VpZ2h0cy5zaGFwZX0uYCk7XG5cbiAgY29uc3QgaW5wdXRzOiBEZW5zZUJpbmNvdW50SW5wdXRzID0ge3g6ICR4LCB3ZWlnaHRzOiAkd2VpZ2h0c307XG4gIGNvbnN0IGF0dHJzOiBEZW5zZUJpbmNvdW50QXR0cnMgPSB7c2l6ZSwgYmluYXJ5T3V0cHV0fTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIERlbnNlQmluY291bnQsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3QgZGVuc2VCaW5jb3VudCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7ZGVuc2VCaW5jb3VudF99KTtcbiJdfQ==