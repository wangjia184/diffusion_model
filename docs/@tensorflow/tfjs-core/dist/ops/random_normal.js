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
import { assertNonNegativeIntegerDimensions } from '../util_base';
import { buffer } from './buffer';
import { op } from './operation';
import { MPRandGauss } from './rand_util';
/**
 * Creates a `tf.Tensor` with values sampled from a normal distribution.
 *
 * ```js
 * tf.randomNormal([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param mean The mean of the normal distribution.
 * @param stdDev The standard deviation of the normal distribution.
 * @param dtype The data type of the output.
 * @param seed The seed for the random number generator.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomNormal_(shape, mean = 0, stdDev = 1, dtype, seed) {
    assertNonNegativeIntegerDimensions(shape);
    if (dtype != null && dtype === 'bool') {
        throw new Error(`Unsupported data type ${dtype}`);
    }
    const randGauss = new MPRandGauss(mean, stdDev, dtype, false /* truncated */, seed);
    const res = buffer(shape, dtype);
    for (let i = 0; i < res.values.length; i++) {
        res.values[i] = randGauss.nextValue();
    }
    return res.toTensor();
}
export const randomNormal = /* @__PURE__ */ op({ randomNormal_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFuZG9tX25vcm1hbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3JhbmRvbV9ub3JtYWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBSUgsT0FBTyxFQUFDLGtDQUFrQyxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBRWhFLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDaEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsV0FBVyxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRXhDOzs7Ozs7Ozs7Ozs7OztHQWNHO0FBQ0gsU0FBUyxhQUFhLENBQ2xCLEtBQWtCLEVBQUUsSUFBSSxHQUFHLENBQUMsRUFBRSxNQUFNLEdBQUcsQ0FBQyxFQUFFLEtBQXlCLEVBQ25FLElBQWE7SUFDZixrQ0FBa0MsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUMxQyxJQUFJLEtBQUssSUFBSSxJQUFJLElBQUssS0FBa0IsS0FBSyxNQUFNLEVBQUU7UUFDbkQsTUFBTSxJQUFJLEtBQUssQ0FBQyx5QkFBeUIsS0FBSyxFQUFFLENBQUMsQ0FBQztLQUNuRDtJQUNELE1BQU0sU0FBUyxHQUNYLElBQUksV0FBVyxDQUFDLElBQUksRUFBRSxNQUFNLEVBQUUsS0FBSyxFQUFFLEtBQUssQ0FBQyxlQUFlLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDdEUsTUFBTSxHQUFHLEdBQUcsTUFBTSxDQUFDLEtBQUssRUFBRSxLQUFLLENBQUMsQ0FBQztJQUNqQyxLQUFLLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsR0FBRyxDQUFDLE1BQU0sQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUU7UUFDMUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsR0FBRyxTQUFTLENBQUMsU0FBUyxFQUFFLENBQUM7S0FDdkM7SUFDRCxPQUFPLEdBQUcsQ0FBQyxRQUFRLEVBQUUsQ0FBQztBQUN4QixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sWUFBWSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxhQUFhLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7RGF0YVR5cGUsIFJhbmssIFNoYXBlTWFwfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge2Fzc2VydE5vbk5lZ2F0aXZlSW50ZWdlckRpbWVuc2lvbnN9IGZyb20gJy4uL3V0aWxfYmFzZSc7XG5cbmltcG9ydCB7YnVmZmVyfSBmcm9tICcuL2J1ZmZlcic7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge01QUmFuZEdhdXNzfSBmcm9tICcuL3JhbmRfdXRpbCc7XG5cbi8qKlxuICogQ3JlYXRlcyBhIGB0Zi5UZW5zb3JgIHdpdGggdmFsdWVzIHNhbXBsZWQgZnJvbSBhIG5vcm1hbCBkaXN0cmlidXRpb24uXG4gKlxuICogYGBganNcbiAqIHRmLnJhbmRvbU5vcm1hbChbMiwgMl0pLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gc2hhcGUgQW4gYXJyYXkgb2YgaW50ZWdlcnMgZGVmaW5pbmcgdGhlIG91dHB1dCB0ZW5zb3Igc2hhcGUuXG4gKiBAcGFyYW0gbWVhbiBUaGUgbWVhbiBvZiB0aGUgbm9ybWFsIGRpc3RyaWJ1dGlvbi5cbiAqIEBwYXJhbSBzdGREZXYgVGhlIHN0YW5kYXJkIGRldmlhdGlvbiBvZiB0aGUgbm9ybWFsIGRpc3RyaWJ1dGlvbi5cbiAqIEBwYXJhbSBkdHlwZSBUaGUgZGF0YSB0eXBlIG9mIHRoZSBvdXRwdXQuXG4gKiBAcGFyYW0gc2VlZCBUaGUgc2VlZCBmb3IgdGhlIHJhbmRvbSBudW1iZXIgZ2VuZXJhdG9yLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ1JhbmRvbSd9XG4gKi9cbmZ1bmN0aW9uIHJhbmRvbU5vcm1hbF88UiBleHRlbmRzIFJhbms+KFxuICAgIHNoYXBlOiBTaGFwZU1hcFtSXSwgbWVhbiA9IDAsIHN0ZERldiA9IDEsIGR0eXBlPzogJ2Zsb2F0MzInfCdpbnQzMicsXG4gICAgc2VlZD86IG51bWJlcik6IFRlbnNvcjxSPiB7XG4gIGFzc2VydE5vbk5lZ2F0aXZlSW50ZWdlckRpbWVuc2lvbnMoc2hhcGUpO1xuICBpZiAoZHR5cGUgIT0gbnVsbCAmJiAoZHR5cGUgYXMgRGF0YVR5cGUpID09PSAnYm9vbCcpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoYFVuc3VwcG9ydGVkIGRhdGEgdHlwZSAke2R0eXBlfWApO1xuICB9XG4gIGNvbnN0IHJhbmRHYXVzcyA9XG4gICAgICBuZXcgTVBSYW5kR2F1c3MobWVhbiwgc3RkRGV2LCBkdHlwZSwgZmFsc2UgLyogdHJ1bmNhdGVkICovLCBzZWVkKTtcbiAgY29uc3QgcmVzID0gYnVmZmVyKHNoYXBlLCBkdHlwZSk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcmVzLnZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgIHJlcy52YWx1ZXNbaV0gPSByYW5kR2F1c3MubmV4dFZhbHVlKCk7XG4gIH1cbiAgcmV0dXJuIHJlcy50b1RlbnNvcigpO1xufVxuXG5leHBvcnQgY29uc3QgcmFuZG9tTm9ybWFsID0gLyogQF9fUFVSRV9fICovIG9wKHtyYW5kb21Ob3JtYWxffSk7XG4iXX0=