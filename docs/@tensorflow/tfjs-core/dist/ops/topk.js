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
import { ENGINE } from '../engine';
import { TopK } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Finds the values and indices of the `k` largest entries along the last
 * dimension.
 *
 * If the input is a vector (rank=1), finds the k largest entries in the vector
 * and outputs their values and indices as vectors. Thus values[j] is the j-th
 * largest entry in input, and its index is indices[j].
 * For higher rank inputs, computes the top k entries along the last dimension.
 *
 * If two elements are equal, the lower-index element appears first.
 *
 * ```js
 * const a = tf.tensor2d([[1, 5], [4, 3]]);
 * const {values, indices} = tf.topk(a);
 * values.print();
 * indices.print();
 * ```
 * @param x 1-D or higher `tf.Tensor` with last dimension being at least `k`.
 * @param k Number of top elements to look for along the last dimension.
 * @param sorted If true, the resulting `k` elements will be sorted by the
 *     values in descending order.
 *
 * @doc {heading: 'Operations', subheading: 'Evaluation'}
 */
function topk_(x, k = 1, sorted = true) {
    const $x = convertToTensor(x, 'x', 'topk');
    if ($x.rank === 0) {
        throw new Error('topk() expects the input to be of rank 1 or higher');
    }
    const lastDim = $x.shape[$x.shape.length - 1];
    if (k < 0) {
        throw new Error(`'k' passed to topk() must be >= 0 but got ${k}`);
    }
    if (k > lastDim) {
        throw new Error(`'k' passed to topk() must be <= the last dimension (${lastDim}) ` +
            `but got ${k}`);
    }
    const inputs = { x: $x };
    const attrs = { k, sorted };
    const [values, indices] = ENGINE.runKernel(TopK, inputs, attrs);
    return { values, indices };
}
export const topk = /* @__PURE__ */ op({ topk_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidG9way5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3RvcGsudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsSUFBSSxFQUF3QixNQUFNLGlCQUFpQixDQUFDO0FBSTVELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXVCRztBQUNILFNBQVMsS0FBSyxDQUNWLENBQWUsRUFBRSxDQUFDLEdBQUcsQ0FBQyxFQUFFLE1BQU0sR0FBRyxJQUFJO0lBQ3ZDLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQzNDLElBQUksRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDakIsTUFBTSxJQUFJLEtBQUssQ0FBQyxvREFBb0QsQ0FBQyxDQUFDO0tBQ3ZFO0lBQ0QsTUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxFQUFFLENBQUMsS0FBSyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQztJQUU5QyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUU7UUFDVCxNQUFNLElBQUksS0FBSyxDQUFDLDZDQUE2QyxDQUFDLEVBQUUsQ0FBQyxDQUFDO0tBQ25FO0lBRUQsSUFBSSxDQUFDLEdBQUcsT0FBTyxFQUFFO1FBQ2YsTUFBTSxJQUFJLEtBQUssQ0FDWCx1REFBdUQsT0FBTyxJQUFJO1lBQ2xFLFdBQVcsQ0FBQyxFQUFFLENBQUMsQ0FBQztLQUNyQjtJQUVELE1BQU0sTUFBTSxHQUFlLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQ25DLE1BQU0sS0FBSyxHQUFjLEVBQUMsQ0FBQyxFQUFFLE1BQU0sRUFBQyxDQUFDO0lBRXJDLE1BQU0sQ0FBQyxNQUFNLEVBQUUsT0FBTyxDQUFDLEdBQUcsTUFBTSxDQUFDLFNBQVMsQ0FDdEMsSUFBSSxFQUFFLE1BQW1DLEVBQ3pDLEtBQWdDLENBQVcsQ0FBQztJQUVoRCxPQUFPLEVBQUMsTUFBTSxFQUFFLE9BQU8sRUFBQyxDQUFDO0FBQzNCLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxJQUFJLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLEtBQUssRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtUb3BLLCBUb3BLQXR0cnMsIFRvcEtJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBGaW5kcyB0aGUgdmFsdWVzIGFuZCBpbmRpY2VzIG9mIHRoZSBga2AgbGFyZ2VzdCBlbnRyaWVzIGFsb25nIHRoZSBsYXN0XG4gKiBkaW1lbnNpb24uXG4gKlxuICogSWYgdGhlIGlucHV0IGlzIGEgdmVjdG9yIChyYW5rPTEpLCBmaW5kcyB0aGUgayBsYXJnZXN0IGVudHJpZXMgaW4gdGhlIHZlY3RvclxuICogYW5kIG91dHB1dHMgdGhlaXIgdmFsdWVzIGFuZCBpbmRpY2VzIGFzIHZlY3RvcnMuIFRodXMgdmFsdWVzW2pdIGlzIHRoZSBqLXRoXG4gKiBsYXJnZXN0IGVudHJ5IGluIGlucHV0LCBhbmQgaXRzIGluZGV4IGlzIGluZGljZXNbal0uXG4gKiBGb3IgaGlnaGVyIHJhbmsgaW5wdXRzLCBjb21wdXRlcyB0aGUgdG9wIGsgZW50cmllcyBhbG9uZyB0aGUgbGFzdCBkaW1lbnNpb24uXG4gKlxuICogSWYgdHdvIGVsZW1lbnRzIGFyZSBlcXVhbCwgdGhlIGxvd2VyLWluZGV4IGVsZW1lbnQgYXBwZWFycyBmaXJzdC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgYSA9IHRmLnRlbnNvcjJkKFtbMSwgNV0sIFs0LCAzXV0pO1xuICogY29uc3Qge3ZhbHVlcywgaW5kaWNlc30gPSB0Zi50b3BrKGEpO1xuICogdmFsdWVzLnByaW50KCk7XG4gKiBpbmRpY2VzLnByaW50KCk7XG4gKiBgYGBcbiAqIEBwYXJhbSB4IDEtRCBvciBoaWdoZXIgYHRmLlRlbnNvcmAgd2l0aCBsYXN0IGRpbWVuc2lvbiBiZWluZyBhdCBsZWFzdCBga2AuXG4gKiBAcGFyYW0gayBOdW1iZXIgb2YgdG9wIGVsZW1lbnRzIHRvIGxvb2sgZm9yIGFsb25nIHRoZSBsYXN0IGRpbWVuc2lvbi5cbiAqIEBwYXJhbSBzb3J0ZWQgSWYgdHJ1ZSwgdGhlIHJlc3VsdGluZyBga2AgZWxlbWVudHMgd2lsbCBiZSBzb3J0ZWQgYnkgdGhlXG4gKiAgICAgdmFsdWVzIGluIGRlc2NlbmRpbmcgb3JkZXIuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnRXZhbHVhdGlvbid9XG4gKi9cbmZ1bmN0aW9uIHRvcGtfPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFR8VGVuc29yTGlrZSwgayA9IDEsIHNvcnRlZCA9IHRydWUpOiB7dmFsdWVzOiBULCBpbmRpY2VzOiBUfSB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ3RvcGsnKTtcbiAgaWYgKCR4LnJhbmsgPT09IDApIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ3RvcGsoKSBleHBlY3RzIHRoZSBpbnB1dCB0byBiZSBvZiByYW5rIDEgb3IgaGlnaGVyJyk7XG4gIH1cbiAgY29uc3QgbGFzdERpbSA9ICR4LnNoYXBlWyR4LnNoYXBlLmxlbmd0aCAtIDFdO1xuXG4gIGlmIChrIDwgMCkge1xuICAgIHRocm93IG5ldyBFcnJvcihgJ2snIHBhc3NlZCB0byB0b3BrKCkgbXVzdCBiZSA+PSAwIGJ1dCBnb3QgJHtrfWApO1xuICB9XG5cbiAgaWYgKGsgPiBsYXN0RGltKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICBgJ2snIHBhc3NlZCB0byB0b3BrKCkgbXVzdCBiZSA8PSB0aGUgbGFzdCBkaW1lbnNpb24gKCR7bGFzdERpbX0pIGAgK1xuICAgICAgICBgYnV0IGdvdCAke2t9YCk7XG4gIH1cblxuICBjb25zdCBpbnB1dHM6IFRvcEtJbnB1dHMgPSB7eDogJHh9O1xuICBjb25zdCBhdHRyczogVG9wS0F0dHJzID0ge2ssIHNvcnRlZH07XG5cbiAgY29uc3QgW3ZhbHVlcywgaW5kaWNlc10gPSBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgVG9wSywgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCkgYXMgW1QsIFRdO1xuXG4gIHJldHVybiB7dmFsdWVzLCBpbmRpY2VzfTtcbn1cblxuZXhwb3J0IGNvbnN0IHRvcGsgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3RvcGtffSk7XG4iXX0=