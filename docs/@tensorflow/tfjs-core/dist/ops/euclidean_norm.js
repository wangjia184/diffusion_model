/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
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
import { norm } from './norm';
import { op } from './operation';
/**
 * Computes the Euclidean norm of scalar, vectors, and matrices.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * x.euclideanNorm().print();  // or tf.euclideanNorm(x)
 * ```
 *
 * @param x The input array.
 * @param axis Optional. If axis is null (the default), the input is
 * considered a vector and a single vector norm is computed over the entire
 * set of values in the Tensor, i.e. euclideanNorm(x) is equivalent
 * to euclideanNorm(x.reshape([-1])). If axis is an integer, the input
 * is considered a batch of vectors, and axis determines the axis in x
 * over which to compute vector norms. If axis is a 2-tuple of integer it is
 * considered a batch of matrices and axis determines the axes in NDArray
 * over which to compute a matrix norm.
 * @param keepDims Optional. If true, the norm has the same dimensionality
 * as the input.
 *
 * @doc {heading: 'Operations', subheading: 'Matrices'}
 */
function euclideanNorm_(x, axis = null, keepDims = false) {
    return norm(x, 'euclidean', axis, keepDims);
}
export const euclideanNorm = /* @__PURE__ */ op({ euclideanNorm_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXVjbGlkZWFuX25vcm0uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9ldWNsaWRlYW5fbm9ybS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFLSCxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sUUFBUSxDQUFDO0FBQzVCLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FzQkc7QUFDSCxTQUFTLGNBQWMsQ0FDbkIsQ0FBb0IsRUFBRSxPQUF3QixJQUFJLEVBQ2xELFFBQVEsR0FBRyxLQUFLO0lBQ2xCLE9BQU8sSUFBSSxDQUFDLENBQUMsRUFBRSxXQUFXLEVBQUUsSUFBSSxFQUFFLFFBQVEsQ0FBQyxDQUFDO0FBQzlDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxhQUFhLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGNBQWMsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7bm9ybX0gZnJvbSAnLi9ub3JtJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgRXVjbGlkZWFuIG5vcm0gb2Ygc2NhbGFyLCB2ZWN0b3JzLCBhbmQgbWF0cmljZXMuXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IxZChbMSwgMiwgMywgNF0pO1xuICpcbiAqIHguZXVjbGlkZWFuTm9ybSgpLnByaW50KCk7ICAvLyBvciB0Zi5ldWNsaWRlYW5Ob3JtKHgpXG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgYXJyYXkuXG4gKiBAcGFyYW0gYXhpcyBPcHRpb25hbC4gSWYgYXhpcyBpcyBudWxsICh0aGUgZGVmYXVsdCksIHRoZSBpbnB1dCBpc1xuICogY29uc2lkZXJlZCBhIHZlY3RvciBhbmQgYSBzaW5nbGUgdmVjdG9yIG5vcm0gaXMgY29tcHV0ZWQgb3ZlciB0aGUgZW50aXJlXG4gKiBzZXQgb2YgdmFsdWVzIGluIHRoZSBUZW5zb3IsIGkuZS4gZXVjbGlkZWFuTm9ybSh4KSBpcyBlcXVpdmFsZW50XG4gKiB0byBldWNsaWRlYW5Ob3JtKHgucmVzaGFwZShbLTFdKSkuIElmIGF4aXMgaXMgYW4gaW50ZWdlciwgdGhlIGlucHV0XG4gKiBpcyBjb25zaWRlcmVkIGEgYmF0Y2ggb2YgdmVjdG9ycywgYW5kIGF4aXMgZGV0ZXJtaW5lcyB0aGUgYXhpcyBpbiB4XG4gKiBvdmVyIHdoaWNoIHRvIGNvbXB1dGUgdmVjdG9yIG5vcm1zLiBJZiBheGlzIGlzIGEgMi10dXBsZSBvZiBpbnRlZ2VyIGl0IGlzXG4gKiBjb25zaWRlcmVkIGEgYmF0Y2ggb2YgbWF0cmljZXMgYW5kIGF4aXMgZGV0ZXJtaW5lcyB0aGUgYXhlcyBpbiBOREFycmF5XG4gKiBvdmVyIHdoaWNoIHRvIGNvbXB1dGUgYSBtYXRyaXggbm9ybS5cbiAqIEBwYXJhbSBrZWVwRGltcyBPcHRpb25hbC4gSWYgdHJ1ZSwgdGhlIG5vcm0gaGFzIHRoZSBzYW1lIGRpbWVuc2lvbmFsaXR5XG4gKiBhcyB0aGUgaW5wdXQuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnTWF0cmljZXMnfVxuICovXG5mdW5jdGlvbiBldWNsaWRlYW5Ob3JtXyhcbiAgICB4OiBUZW5zb3J8VGVuc29yTGlrZSwgYXhpczogbnVtYmVyfG51bWJlcltdID0gbnVsbCxcbiAgICBrZWVwRGltcyA9IGZhbHNlKTogVGVuc29yIHtcbiAgcmV0dXJuIG5vcm0oeCwgJ2V1Y2xpZGVhbicsIGF4aXMsIGtlZXBEaW1zKTtcbn1cblxuZXhwb3J0IGNvbnN0IGV1Y2xpZGVhbk5vcm0gPSAvKiBAX19QVVJFX18gKi8gb3Aoe2V1Y2xpZGVhbk5vcm1ffSk7XG4iXX0=