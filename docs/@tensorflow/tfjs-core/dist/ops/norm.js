/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { abs } from './abs';
import * as axis_util from './axis_util';
import { max } from './max';
import { min } from './min';
import { op } from './operation';
import { pow } from './pow';
import { reshape } from './reshape';
import { scalar } from './scalar';
import { sqrt } from './sqrt';
import { square } from './square';
import { sum } from './sum';
/**
 * Computes the norm of scalar, vectors, and matrices.
 * This function can compute several different vector norms (the 1-norm, the
 * Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0)
 * and matrix norms (Frobenius, 1-norm, and inf-norm).
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * x.norm().print();  // or tf.norm(x)
 * ```
 *
 * @param x The input array.
 * @param ord Optional. Order of the norm. Supported norm types are
 * following:
 *
 *  | ord        | norm for matrices         | norm for vectors
 *  |------------|---------------------------|---------------------
 *  |'euclidean' |Frobenius norm             |2-norm
 *  |'fro'       |Frobenius norm	           |
 *  |Infinity    |max(sum(abs(x), axis=1))   |max(abs(x))
 *  |-Infinity   |min(sum(abs(x), axis=1))   |min(abs(x))
 *  |1           |max(sum(abs(x), axis=0))   |sum(abs(x))
 *  |2           |                           |sum(abs(x)^2)^(1/2)
 *
 * @param axis Optional. If axis is null (the default), the input is
 * considered a vector and a single vector norm is computed over the entire
 * set of values in the Tensor, i.e. norm(x, ord) is equivalent
 * to norm(x.reshape([-1]), ord). If axis is an integer, the input
 * is considered a batch of vectors, and axis determines the axis in x
 * over which to compute vector norms. If axis is a 2-tuple of integer it is
 * considered a batch of matrices and axis determines the axes in NDArray
 * over which to compute a matrix norm.
 * @param keepDims Optional. If true, the norm has the same dimensionality
 * as the input.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function norm_(x, ord = 'euclidean', axis = null, keepDims = false) {
    x = convertToTensor(x, 'x', 'norm');
    const norm = normImpl(x, ord, axis);
    let keepDimsShape = norm.shape;
    if (keepDims) {
        const axes = parseAxisParam(axis, x.shape);
        keepDimsShape = axis_util.expandShapeToKeepDim(norm.shape, axes);
    }
    return reshape(norm, keepDimsShape);
}
function normImpl(x, p, axis = null) {
    if (x.rank === 0) {
        return abs(x);
    }
    // consider vector when no axis is specified
    if (x.rank !== 1 && axis === null) {
        return normImpl(reshape(x, [-1]), p, axis);
    }
    // vector
    if (x.rank === 1 || typeof axis === 'number' ||
        Array.isArray(axis) && axis.length === 1) {
        if (p === 1) {
            return sum(abs(x), axis);
        }
        if (p === Infinity) {
            return max(abs(x), axis);
        }
        if (p === -Infinity) {
            return min(abs(x), axis);
        }
        if (p === 'euclidean' || p === 2) {
            // norm(x, 2) = sum(abs(xi) ^ 2) ^ 1/2
            return sqrt(sum(pow(abs(x), scalar(2, 'int32')), axis));
        }
        throw new Error(`Error in norm: invalid ord value: ${p}`);
    }
    // matrix (assumption axis[0] < axis[1])
    if (Array.isArray(axis) && axis.length === 2) {
        if (p === 1) {
            return max(sum(abs(x), axis[0]), axis[1] - 1);
        }
        if (p === Infinity) {
            return max(sum(abs(x), axis[1]), axis[0]);
        }
        if (p === -Infinity) {
            return min(sum(abs(x), axis[1]), axis[0]);
        }
        if (p === 'fro' || p === 'euclidean') {
            // norm(x) = sqrt(sum(pow(x, 2)))
            return sqrt(sum(square(x), axis));
        }
        throw new Error(`Error in norm: invalid ord value: ${p}`);
    }
    throw new Error(`Error in norm: invalid axis: ${axis}`);
}
export const norm = /* @__PURE__ */ op({ norm_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibm9ybS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL25vcm0udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sRUFBQyxjQUFjLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFdkMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUMxQixPQUFPLEtBQUssU0FBUyxNQUFNLGFBQWEsQ0FBQztBQUN6QyxPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxPQUFPLENBQUM7QUFDMUIsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUMvQixPQUFPLEVBQUMsR0FBRyxFQUFDLE1BQU0sT0FBTyxDQUFDO0FBQzFCLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEMsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUNoQyxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzVCLE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDaEMsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLE9BQU8sQ0FBQztBQUUxQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXFDRztBQUNILFNBQVMsS0FBSyxDQUNWLENBQW9CLEVBQUUsTUFBZ0MsV0FBVyxFQUNqRSxPQUF3QixJQUFJLEVBQUUsUUFBUSxHQUFHLEtBQUs7SUFDaEQsQ0FBQyxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBRXBDLE1BQU0sSUFBSSxHQUFHLFFBQVEsQ0FBQyxD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QUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FDM0M7UUFDRCxJQUFJLENBQUMsS0FBSyxLQUFLLElBQUksQ0FBQyxLQUFLLFdBQVcsRUFBRTtZQUNwQyxpQ0FBaUM7WUFDakMsT0FBTyxJQUFJLENBQUMsR0FBRyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO1NBQ25DO1FBRUQsTUFBTSxJQUFJLEtBQUssQ0FBQyxxQ0FBcUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztLQUMzRDtJQUVELE1BQU0sSUFBSSxLQUFLLENBQUMsZ0NBQWdDLElBQUksRUFBRSxDQUFDLENBQUM7QUFDMUQsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLElBQUksR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsS0FBSyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3J9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuaW1wb3J0IHtwYXJzZUF4aXNQYXJhbX0gZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7YWJzfSBmcm9tICcuL2Ficyc7XG5pbXBvcnQgKiBhcyBheGlzX3V0aWwgZnJvbSAnLi9heGlzX3V0aWwnO1xuaW1wb3J0IHttYXh9IGZyb20gJy4vbWF4JztcbmltcG9ydCB7bWlufSBmcm9tICcuL21pbic7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Bvd30gZnJvbSAnLi9wb3cnO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuL3Jlc2hhcGUnO1xuaW1wb3J0IHtzY2FsYXJ9IGZyb20gJy4vc2NhbGFyJztcbmltcG9ydCB7c3FydH0gZnJvbSAnLi9zcXJ0JztcbmltcG9ydCB7c3F1YXJlfSBmcm9tICcuL3NxdWFyZSc7XG5pbXBvcnQge3N1bX0gZnJvbSAnLi9zdW0nO1xuXG4vKipcbiAqIENvbXB1dGVzIHRoZSBub3JtIG9mIHNjYWxhciwgdmVjdG9ycywgYW5kIG1hdHJpY2VzLlxuICogVGhpcyBmdW5jdGlvbiBjYW4gY29tcHV0ZSBzZXZlcmFsIGRpZmZlcmVudCB2ZWN0b3Igbm9ybXMgKHRoZSAxLW5vcm0sIHRoZVxuICogRXVjbGlkZWFuIG9yIDItbm9ybSwgdGhlIGluZi1ub3JtLCBhbmQgaW4gZ2VuZXJhbCB0aGUgcC1ub3JtIGZvciBwID4gMClcbiAqIGFuZCBtYXRyaXggbm9ybXMgKEZyb2Jlbml1cywgMS1ub3JtLCBhbmQgaW5mLW5vcm0pLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDMsIDRdKTtcbiAqXG4gKiB4Lm5vcm0oKS5wcmludCgpOyAgLy8gb3IgdGYubm9ybSh4KVxuICogYGBgXG4gKlxuICogQHBhcmFtIHggVGhlIGlucHV0IGFycmF5LlxuICogQHBhcmFtIG9yZCBPcHRpb25hbC4gT3JkZXIgb2YgdGhlIG5vcm0uIFN1cHBvcnRlZCBub3JtIHR5cGVzIGFyZVxuICogZm9sbG93aW5nOlxuICpcbiAqICB8IG9yZCAgICAgICAgfCBub3JtIGZvciBtYXRyaWNlcyAgICAgICAgIHwgbm9ybSBmb3IgdmVjdG9yc1xuICogIHwtLS0tLS0tLS0tLS18LS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tfC0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICogIHwnZXVjbGlkZWFuJyB8RnJvYmVuaXVzIG5vcm0gICAgICAgICAgICAgfDItbm9ybVxuICogIHwnZnJvJyAgICAgICB8RnJvYmVuaXVzIG5vcm1cdCAgICAgICAgICAgfFxuICogIHxJbmZpbml0eSAgICB8bWF4KHN1bShhYnMoeCksIGF4aXM9MSkpICAgfG1heChhYnMoeCkpXG4gKiAgfC1JbmZpbml0eSAgIHxtaW4oc3VtKGFicyh4KSwgYXhpcz0xKSkgICB8bWluKGFicyh4KSlcbiAqICB8MSAgICAgICAgICAgfG1heChzdW0oYWJzKHgpLCBheGlzPTApKSAgIHxzdW0oYWJzKHgpKVxuICogIHwyICAgICAgICAgICB8ICAgICAgICAgICAgICAgICAgICAgICAgICAgfHN1bShhYnMoeCleMileKDEvMilcbiAqXG4gKiBAcGFyYW0gYXhpcyBPcHRpb25hbC4gSWYgYXhpcyBpcyBudWxsICh0aGUgZGVmYXVsdCksIHRoZSBpbnB1dCBpc1xuICogY29uc2lkZXJlZCBhIHZlY3RvciBhbmQgYSBzaW5nbGUgdmVjdG9yIG5vcm0gaXMgY29tcHV0ZWQgb3ZlciB0aGUgZW50aXJlXG4gKiBzZXQgb2YgdmFsdWVzIGluIHRoZSBUZW5zb3IsIGkuZS4gbm9ybSh4LCBvcmQpIGlzIGVxdWl2YWxlbnRcbiAqIHRvIG5vcm0oeC5yZXNoYXBlKFstMV0pLCBvcmQpLiBJZiBheGlzIGlzIGFuIGludGVnZXIsIHRoZSBpbnB1dFxuICogaXMgY29uc2lkZXJlZCBhIGJhdGNoIG9mIHZlY3RvcnMsIGFuZCBheGlzIGRldGVybWluZXMgdGhlIGF4aXMgaW4geFxuICogb3ZlciB3aGljaCB0byBjb21wdXRlIHZlY3RvciBub3Jtcy4gSWYgYXhpcyBpcyBhIDItdHVwbGUgb2YgaW50ZWdlciBpdCBpc1xuICogY29uc2lkZXJlZCBhIGJhdGNoIG9mIG1hdHJpY2VzIGFuZCBheGlzIGRldGVybWluZXMgdGhlIGF4ZXMgaW4gTkRBcnJheVxuICogb3ZlciB3aGljaCB0byBjb21wdXRlIGEgbWF0cml4IG5vcm0uXG4gKiBAcGFyYW0ga2VlcERpbXMgT3B0aW9uYWwuIElmIHRydWUsIHRoZSBub3JtIGhhcyB0aGUgc2FtZSBkaW1lbnNpb25hbGl0eVxuICogYXMgdGhlIGlucHV0LlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ01hdHJpY2VzJ31cbiAqL1xuZnVuY3Rpb24gbm9ybV8oXG4gICAgeDogVGVuc29yfFRlbnNvckxpa2UsIG9yZDogbnVtYmVyfCdldWNsaWRlYW4nfCdmcm8nID0gJ2V1Y2xpZGVhbicsXG4gICAgYXhpczogbnVtYmVyfG51bWJlcltdID0gbnVsbCwga2VlcERpbXMgPSBmYWxzZSk6IFRlbnNvciB7XG4gIHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnbm9ybScpO1xuXG4gIGNvbnN0IG5vcm0gPSBub3JtSW1wbCh4LCBvcmQsIGF4aXMpO1xuICBsZXQga2VlcERpbXNTaGFwZSA9IG5vcm0uc2hhcGU7XG4gIGlmIChrZWVwRGltcykge1xuICAgIGNvbnN0IGF4ZXMgPSBwYXJzZUF4aXNQYXJhbShheGlzLCB4LnNoYXBlKTtcbiAgICBrZWVwRGltc1NoYXBlID0gYXhpc191dGlsLmV4cGFuZFNoYXBlVG9LZWVwRGltKG5vcm0uc2hhcGUsIGF4ZXMpO1xuICB9XG4gIHJldHVybiByZXNoYXBlKG5vcm0sIGtlZXBEaW1zU2hhcGUpO1xufVxuXG5mdW5jdGlvbiBub3JtSW1wbChcbiAgICB4OiBUZW5zb3IsIHA6IG51bWJlcnxzdHJpbmcsIGF4aXM6IG51bWJlcnxudW1iZXJbXSA9IG51bGwpOiBUZW5zb3Ige1xuICBpZiAoeC5yYW5rID09PSAwKSB7XG4gICAgcmV0dXJuIGFicyh4KTtcbiAgfVxuXG4gIC8vIGNvbnNpZGVyIHZlY3RvciB3aGVuIG5vIGF4aXMgaXMgc3BlY2lmaWVkXG4gIGlmICh4LnJhbmsgIT09IDEgJiYgYXhpcyA9PT0gbnVsbCkge1xuICAgIHJldHVybiBub3JtSW1wbChyZXNoYXBlKHgsIFstMV0pLCBwLCBheGlzKTtcbiAgfVxuXG4gIC8vIHZlY3RvclxuICBpZiAoeC5yYW5rID09PSAxIHx8IHR5cGVvZiBheGlzID09PSAnbnVtYmVyJyB8fFxuICAgICAgQXJyYXkuaXNBcnJheShheGlzKSAmJiBheGlzLmxlbmd0aCA9PT0gMSkge1xuICAgIGlmIChwID09PSAxKSB7XG4gICAgICByZXR1cm4gc3VtKGFicyh4KSwgYXhpcyk7XG4gICAgfVxuICAgIGlmIChwID09PSBJbmZpbml0eSkge1xuICAgICAgcmV0dXJuIG1heChhYnMoeCksIGF4aXMpO1xuICAgIH1cbiAgICBpZiAocCA9PT0gLUluZmluaXR5KSB7XG4gICAgICByZXR1cm4gbWluKGFicyh4KSwgYXhpcyk7XG4gICAgfVxuICAgIGlmIChwID09PSAnZXVjbGlkZWFuJyB8fCBwID09PSAyKSB7XG4gICAgICAvLyBub3JtKHgsIDIpID0gc3VtKGFicyh4aSkgXiAyKSBeIDEvMlxuICAgICAgcmV0dXJuIHNxcnQoc3VtKHBvdyhhYnMoeCksIHNjYWxhcigyLCAnaW50MzInKSksIGF4aXMpKTtcbiAgICB9XG5cbiAgICB0aHJvdyBuZXcgRXJyb3IoYEVycm9yIGluIG5vcm06IGludmFsaWQgb3JkIHZhbHVlOiAke3B9YCk7XG4gIH1cblxuICAvLyBtYXRyaXggKGFzc3VtcHRpb24gYXhpc1swXSA8IGF4aXNbMV0pXG4gIGlmIChBcnJheS5pc0FycmF5KGF4aXMpICYmIGF4aXMubGVuZ3RoID09PSAyKSB7XG4gICAgaWYgKHAgPT09IDEpIHtcbiAgICAgIHJldHVybiBtYXgoc3VtKGFicyh4KSwgYXhpc1swXSksIGF4aXNbMV0gLSAxKTtcbiAgICB9XG4gICAgaWYgKHAgPT09IEluZmluaXR5KSB7XG4gICAgICByZXR1cm4gbWF4KHN1bShhYnMoeCksIGF4aXNbMV0pLCBheGlzWzBdKTtcbiAgICB9XG4gICAgaWYgKHAgPT09IC1JbmZpbml0eSkge1xuICAgICAgcmV0dXJuIG1pbihzdW0oYWJzKHgpLCBheGlzWzFdKSwgYXhpc1swXSk7XG4gICAgfVxuICAgIGlmIChwID09PSAnZnJvJyB8fCBwID09PSAnZXVjbGlkZWFuJykge1xuICAgICAgLy8gbm9ybSh4KSA9IHNxcnQoc3VtKHBvdyh4LCAyKSkpXG4gICAgICByZXR1cm4gc3FydChzdW0oc3F1YXJlKHgpLCBheGlzKSk7XG4gICAgfVxuXG4gICAgdGhyb3cgbmV3IEVycm9yKGBFcnJvciBpbiBub3JtOiBpbnZhbGlkIG9yZCB2YWx1ZTogJHtwfWApO1xuICB9XG5cbiAgdGhyb3cgbmV3IEVycm9yKGBFcnJvciBpbiBub3JtOiBpbnZhbGlkIGF4aXM6ICR7YXhpc31gKTtcbn1cblxuZXhwb3J0IGNvbnN0IG5vcm0gPSAvKiBAX19QVVJFX18gKi8gb3Aoe25vcm1ffSk7XG4iXX0=