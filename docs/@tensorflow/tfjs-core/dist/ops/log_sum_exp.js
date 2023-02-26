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
import { convertToTensor } from '../tensor_util_env';
import { parseAxisParam } from '../util';
import { add } from './add';
import { expandShapeToKeepDim } from './axis_util';
import { exp } from './exp';
import { log } from './log';
import { max } from './max';
import { op } from './operation';
import { reshape } from './reshape';
import { sub } from './sub';
import { sum } from './sum';
/**
 * Computes the log(sum(exp(elements across the reduction dimensions))).
 *
 * Reduces the input along the dimensions given in `axis`. Unless `keepDims`
 * is true, the rank of the array is reduced by 1 for each entry in `axis`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axis` has no entries, all dimensions are reduced, and an array with a
 * single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.logSumExp().print();  // or tf.logSumExp(x)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.logSumExp(axis).print();  // or tf.logSumExp(a, axis)
 * ```
 * @param x The input tensor.
 * @param axis The dimension(s) to reduce. If null (the default),
 *     reduces all dimensions.
 * @param keepDims If true, retains reduced dimensions with length
 *     of 1. Defaults to false.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function logSumExp_(x, axis = null, keepDims = false) {
    const $x = convertToTensor(x, 'x', 'logSumExp');
    const axes = parseAxisParam(axis, $x.shape);
    const xMax = max($x, axes, true /* keepDims */);
    const a = sub($x, xMax);
    const b = exp(a);
    const c = sum(b, axes);
    const d = log(c);
    const res = add(reshape(xMax, d.shape), d);
    if (keepDims) {
        const newShape = expandShapeToKeepDim(res.shape, axes);
        return reshape(res, newShape);
    }
    return res;
}
export const logSumExp = /* @__PURE__ */ op({ logSumExp_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibG9nX3N1bV9leHAuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9sb2dfc3VtX2V4cC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFHSCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxFQUFDLGNBQWMsRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUV2QyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxvQkFBb0IsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUNqRCxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBRTFCOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBNEJHO0FBQ0gsU0FBUyxVQUFVLENBQ2YsQ0FBb0IsRUFBRSxPQUF3QixJQUFJLEVBQUUsUUFBUSxHQUFHLEtBQUs7SUFDdEUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFFaEQsTUFBTSxJQUFJLEdBQUcsY0FBYyxDQUFDLElBQUksRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDNUMsTUFBTSxJQUFJLEdBQUcsR0FBRyxDQUFDLEVBQUUsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDO0lBQ2hELE1BQU0sQ0FBQyxHQUFHLEdBQUcsQ0FBQyxFQUFFLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDeEIsTUFBTSxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2pCLE1BQU0sQ0FBQyxHQUFHLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDdkIsTUFBTSxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2pCLE1BQU0sR0FBRyxHQUFHLEdBQUcsQ0FBQyxPQUFPLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUUzQyxJQUFJLFFBQVEsRUFBRTtRQUNaLE1BQU0sUUFBUSxHQUFHLG9CQUFvQixDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFDdkQsT0FBTyxPQUFPLENBQUMsR0FBRyxFQUFFLFFBQVEsQ0FBTSxDQUFDO0tBQ3BDO0lBQ0QsT0FBTyxHQUFRLENBQUM7QUFDbEIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFNBQVMsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsVUFBVSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHtwYXJzZUF4aXNQYXJhbX0gZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7YWRkfSBmcm9tICcuL2FkZCc7XG5pbXBvcnQge2V4cGFuZFNoYXBlVG9LZWVwRGltfSBmcm9tICcuL2F4aXNfdXRpbCc7XG5pbXBvcnQge2V4cH0gZnJvbSAnLi9leHAnO1xuaW1wb3J0IHtsb2d9IGZyb20gJy4vbG9nJztcbmltcG9ydCB7bWF4fSBmcm9tICcuL21heCc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5pbXBvcnQge3N1Yn0gZnJvbSAnLi9zdWInO1xuaW1wb3J0IHtzdW19IGZyb20gJy4vc3VtJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgbG9nKHN1bShleHAoZWxlbWVudHMgYWNyb3NzIHRoZSByZWR1Y3Rpb24gZGltZW5zaW9ucykpKS5cbiAqXG4gKiBSZWR1Y2VzIHRoZSBpbnB1dCBhbG9uZyB0aGUgZGltZW5zaW9ucyBnaXZlbiBpbiBgYXhpc2AuIFVubGVzcyBga2VlcERpbXNgXG4gKiBpcyB0cnVlLCB0aGUgcmFuayBvZiB0aGUgYXJyYXkgaXMgcmVkdWNlZCBieSAxIGZvciBlYWNoIGVudHJ5IGluIGBheGlzYC5cbiAqIElmIGBrZWVwRGltc2AgaXMgdHJ1ZSwgdGhlIHJlZHVjZWQgZGltZW5zaW9ucyBhcmUgcmV0YWluZWQgd2l0aCBsZW5ndGggMS5cbiAqIElmIGBheGlzYCBoYXMgbm8gZW50cmllcywgYWxsIGRpbWVuc2lvbnMgYXJlIHJlZHVjZWQsIGFuZCBhbiBhcnJheSB3aXRoIGFcbiAqIHNpbmdsZSBlbGVtZW50IGlzIHJldHVybmVkLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDNdKTtcbiAqXG4gKiB4LmxvZ1N1bUV4cCgpLnByaW50KCk7ICAvLyBvciB0Zi5sb2dTdW1FeHAoeClcbiAqIGBgYFxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMmQoWzEsIDIsIDMsIDRdLCBbMiwgMl0pO1xuICpcbiAqIGNvbnN0IGF4aXMgPSAxO1xuICogeC5sb2dTdW1FeHAoYXhpcykucHJpbnQoKTsgIC8vIG9yIHRmLmxvZ1N1bUV4cChhLCBheGlzKVxuICogYGBgXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yLlxuICogQHBhcmFtIGF4aXMgVGhlIGRpbWVuc2lvbihzKSB0byByZWR1Y2UuIElmIG51bGwgKHRoZSBkZWZhdWx0KSxcbiAqICAgICByZWR1Y2VzIGFsbCBkaW1lbnNpb25zLlxuICogQHBhcmFtIGtlZXBEaW1zIElmIHRydWUsIHJldGFpbnMgcmVkdWNlZCBkaW1lbnNpb25zIHdpdGggbGVuZ3RoXG4gKiAgICAgb2YgMS4gRGVmYXVsdHMgdG8gZmFsc2UuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnUmVkdWN0aW9uJ31cbiAqL1xuZnVuY3Rpb24gbG9nU3VtRXhwXzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICB4OiBUZW5zb3J8VGVuc29yTGlrZSwgYXhpczogbnVtYmVyfG51bWJlcltdID0gbnVsbCwga2VlcERpbXMgPSBmYWxzZSk6IFQge1xuICBjb25zdCAkeCA9IGNvbnZlcnRUb1RlbnNvcih4LCAneCcsICdsb2dTdW1FeHAnKTtcblxuICBjb25zdCBheGVzID0gcGFyc2VBeGlzUGFyYW0oYXhpcywgJHguc2hhcGUpO1xuICBjb25zdCB4TWF4ID0gbWF4KCR4LCBheGVzLCB0cnVlIC8qIGtlZXBEaW1zICovKTtcbiAgY29uc3QgYSA9IHN1YigkeCwgeE1heCk7XG4gIGNvbnN0IGIgPSBleHAoYSk7XG4gIGNvbnN0IGMgPSBzdW0oYiwgYXhlcyk7XG4gIGNvbnN0IGQgPSBsb2coYyk7XG4gIGNvbnN0IHJlcyA9IGFkZChyZXNoYXBlKHhNYXgsIGQuc2hhcGUpLCBkKTtcblxuICBpZiAoa2VlcERpbXMpIHtcbiAgICBjb25zdCBuZXdTaGFwZSA9IGV4cGFuZFNoYXBlVG9LZWVwRGltKHJlcy5zaGFwZSwgYXhlcyk7XG4gICAgcmV0dXJuIHJlc2hhcGUocmVzLCBuZXdTaGFwZSkgYXMgVDtcbiAgfVxuICByZXR1cm4gcmVzIGFzIFQ7XG59XG5cbmV4cG9ydCBjb25zdCBsb2dTdW1FeHAgPSAvKiBAX19QVVJFX18gKi8gb3Aoe2xvZ1N1bUV4cF99KTtcbiJdfQ==