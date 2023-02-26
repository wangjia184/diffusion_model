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
import { UniformRandom } from './rand_util';
/**
 * Creates a `tf.Tensor` with values sampled from a uniform distribution.
 *
 * The generated values follow a uniform distribution in the range [minval,
 * maxval). The lower bound minval is included in the range, while the upper
 * bound maxval is excluded.
 *
 * ```js
 * tf.randomUniform([2, 2]).print();
 * ```
 *
 * @param shape An array of integers defining the output tensor shape.
 * @param minval The lower bound on the range of random values to generate.
 *   Defaults to 0.
 * @param maxval The upper bound on the range of random values to generate.
 *   Defaults to 1.
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Random'}
 */
function randomUniform_(shape, minval = 0, maxval = 1, dtype = 'float32', seed) {
    assertNonNegativeIntegerDimensions(shape);
    const res = buffer(shape, dtype);
    const random = new UniformRandom(minval, maxval, null, seed);
    for (let i = 0; i < res.values.length; i++) {
        res.values[i] = random.nextValue();
    }
    return res.toTensor();
}
export const randomUniform = /* @__PURE__ */ op({ randomUniform_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFuZG9tX3VuaWZvcm0uanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9yYW5kb21fdW5pZm9ybS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFJSCxPQUFPLEVBQUMsa0NBQWtDLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFFaEUsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFVBQVUsQ0FBQztBQUNoQyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFMUM7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FtQkc7QUFDSCxTQUFTLGNBQWMsQ0FDbkIsS0FBa0IsRUFBRSxNQUFNLEdBQUcsQ0FBQyxFQUFFLE1BQU0sR0FBRyxDQUFDLEVBQUUsUUFBa0IsU0FBUyxFQUN2RSxJQUFvQjtJQUN0QixrQ0FBa0MsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUMxQyxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsS0FBSyxFQUFFLEtBQUssQ0FBQyxDQUFDO0lBQ2pDLE1BQU0sTUFBTSxHQUFHLElBQUksYUFBYSxDQUFDLE1BQU0sRUFBRSxNQUFNLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzdELEtBQUssSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxHQUFHLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRTtRQUMxQyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxTQUFTLEVBQUUsQ0FBQztLQUNwQztJQUNELE9BQU8sR0FBRyxDQUFDLFFBQVEsRUFBRSxDQUFDO0FBQ3hCLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxhQUFhLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLGNBQWMsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtEYXRhVHlwZSwgUmFuaywgU2hhcGVNYXB9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCB7YXNzZXJ0Tm9uTmVnYXRpdmVJbnRlZ2VyRGltZW5zaW9uc30gZnJvbSAnLi4vdXRpbF9iYXNlJztcblxuaW1wb3J0IHtidWZmZXJ9IGZyb20gJy4vYnVmZmVyJztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7VW5pZm9ybVJhbmRvbX0gZnJvbSAnLi9yYW5kX3V0aWwnO1xuXG4vKipcbiAqIENyZWF0ZXMgYSBgdGYuVGVuc29yYCB3aXRoIHZhbHVlcyBzYW1wbGVkIGZyb20gYSB1bmlmb3JtIGRpc3RyaWJ1dGlvbi5cbiAqXG4gKiBUaGUgZ2VuZXJhdGVkIHZhbHVlcyBmb2xsb3cgYSB1bmlmb3JtIGRpc3RyaWJ1dGlvbiBpbiB0aGUgcmFuZ2UgW21pbnZhbCxcbiAqIG1heHZhbCkuIFRoZSBsb3dlciBib3VuZCBtaW52YWwgaXMgaW5jbHVkZWQgaW4gdGhlIHJhbmdlLCB3aGlsZSB0aGUgdXBwZXJcbiAqIGJvdW5kIG1heHZhbCBpcyBleGNsdWRlZC5cbiAqXG4gKiBgYGBqc1xuICogdGYucmFuZG9tVW5pZm9ybShbMiwgMl0pLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gc2hhcGUgQW4gYXJyYXkgb2YgaW50ZWdlcnMgZGVmaW5pbmcgdGhlIG91dHB1dCB0ZW5zb3Igc2hhcGUuXG4gKiBAcGFyYW0gbWludmFsIFRoZSBsb3dlciBib3VuZCBvbiB0aGUgcmFuZ2Ugb2YgcmFuZG9tIHZhbHVlcyB0byBnZW5lcmF0ZS5cbiAqICAgRGVmYXVsdHMgdG8gMC5cbiAqIEBwYXJhbSBtYXh2YWwgVGhlIHVwcGVyIGJvdW5kIG9uIHRoZSByYW5nZSBvZiByYW5kb20gdmFsdWVzIHRvIGdlbmVyYXRlLlxuICogICBEZWZhdWx0cyB0byAxLlxuICogQHBhcmFtIGR0eXBlIFRoZSBkYXRhIHR5cGUgb2YgdGhlIG91dHB1dCB0ZW5zb3IuIERlZmF1bHRzIHRvICdmbG9hdDMyJy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdSYW5kb20nfVxuICovXG5mdW5jdGlvbiByYW5kb21Vbmlmb3JtXzxSIGV4dGVuZHMgUmFuaz4oXG4gICAgc2hhcGU6IFNoYXBlTWFwW1JdLCBtaW52YWwgPSAwLCBtYXh2YWwgPSAxLCBkdHlwZTogRGF0YVR5cGUgPSAnZmxvYXQzMicsXG4gICAgc2VlZD86IG51bWJlcnxzdHJpbmcpOiBUZW5zb3I8Uj4ge1xuICBhc3NlcnROb25OZWdhdGl2ZUludGVnZXJEaW1lbnNpb25zKHNoYXBlKTtcbiAgY29uc3QgcmVzID0gYnVmZmVyKHNoYXBlLCBkdHlwZSk7XG4gIGNvbnN0IHJhbmRvbSA9IG5ldyBVbmlmb3JtUmFuZG9tKG1pbnZhbCwgbWF4dmFsLCBudWxsLCBzZWVkKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCByZXMudmFsdWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgcmVzLnZhbHVlc1tpXSA9IHJhbmRvbS5uZXh0VmFsdWUoKTtcbiAgfVxuICByZXR1cm4gcmVzLnRvVGVuc29yKCk7XG59XG5cbmV4cG9ydCBjb25zdCByYW5kb21Vbmlmb3JtID0gLyogQF9fUFVSRV9fICovIG9wKHtyYW5kb21Vbmlmb3JtX30pO1xuIl19