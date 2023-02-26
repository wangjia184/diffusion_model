/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
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
import { Mean } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Computes the mean of elements across dimensions of a `tf.Tensor`.
 *
 * Reduces `x` along the dimensions given in `axis`. Unless `keepDims` is
 * true, the rank of the `tf.Tensor` is reduced by 1 for each entry in `axis`.
 * If `keepDims` is true, the reduced dimensions are retained with length 1.
 * If `axis` has no entries, all dimensions are reduced, and a `tf.Tensor` with
 * a single element is returned.
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3]);
 *
 * x.mean().print();  // or tf.mean(a)
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * const axis = 1;
 * x.mean(axis).print();  // or tf.mean(x, axis)
 * ```
 *
 * @param x The input tensor.
 * @param axis The dimension(s) to reduce. By default it reduces
 *     all dimensions.
 * @param keepDims If true, retains reduced dimensions with size 1.
 *
 * @doc {heading: 'Operations', subheading: 'Reduction'}
 */
function mean_(x, axis = null, keepDims = false) {
    const $x = convertToTensor(x, 'x', 'mean');
    const inputs = { x: $x };
    const attrs = { axis, keepDims };
    return ENGINE.runKernel(Mean, inputs, attrs);
}
export const mean = /* @__PURE__ */ op({ mean_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibWVhbi5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL21lYW4udHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsSUFBSSxFQUF3QixNQUFNLGlCQUFpQixDQUFDO0FBSTVELE9BQU8sRUFBQyxlQUFlLEVBQUMsTUFBTSxvQkFBb0IsQ0FBQztBQUduRCxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBRS9COzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBNEJHO0FBQ0gsU0FBUyxLQUFLLENBQ1YsQ0FBb0IsRUFBRSxPQUF3QixJQUFJLEVBQUUsUUFBUSxHQUFHLEtBQUs7SUFDdEUsTUFBTSxFQUFFLEdBQUcsZUFBZSxDQUFDLENBQUMsRUFBRSxHQUFHLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFFM0MsTUFBTSxNQUFNLEdBQWUsRUFBQyxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDbkMsTUFBTSxLQUFLLEdBQWMsRUFBQyxJQUFJLEVBQUUsUUFBUSxFQUFDLENBQUM7SUFFMUMsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUNuQixJQUFJLEVBQUUsTUFBbUMsRUFDekMsS0FBZ0MsQ0FBQyxDQUFDO0FBQ3hDLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxJQUFJLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLEtBQUssRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtNZWFuLCBNZWFuQXR0cnMsIE1lYW5JbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb21wdXRlcyB0aGUgbWVhbiBvZiBlbGVtZW50cyBhY3Jvc3MgZGltZW5zaW9ucyBvZiBhIGB0Zi5UZW5zb3JgLlxuICpcbiAqIFJlZHVjZXMgYHhgIGFsb25nIHRoZSBkaW1lbnNpb25zIGdpdmVuIGluIGBheGlzYC4gVW5sZXNzIGBrZWVwRGltc2AgaXNcbiAqIHRydWUsIHRoZSByYW5rIG9mIHRoZSBgdGYuVGVuc29yYCBpcyByZWR1Y2VkIGJ5IDEgZm9yIGVhY2ggZW50cnkgaW4gYGF4aXNgLlxuICogSWYgYGtlZXBEaW1zYCBpcyB0cnVlLCB0aGUgcmVkdWNlZCBkaW1lbnNpb25zIGFyZSByZXRhaW5lZCB3aXRoIGxlbmd0aCAxLlxuICogSWYgYGF4aXNgIGhhcyBubyBlbnRyaWVzLCBhbGwgZGltZW5zaW9ucyBhcmUgcmVkdWNlZCwgYW5kIGEgYHRmLlRlbnNvcmAgd2l0aFxuICogYSBzaW5nbGUgZWxlbWVudCBpcyByZXR1cm5lZC5cbiAqXG4gKiBgYGBqc1xuICogY29uc3QgeCA9IHRmLnRlbnNvcjFkKFsxLCAyLCAzXSk7XG4gKlxuICogeC5tZWFuKCkucHJpbnQoKTsgIC8vIG9yIHRmLm1lYW4oYSlcbiAqIGBgYFxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMmQoWzEsIDIsIDMsIDRdLCBbMiwgMl0pO1xuICpcbiAqIGNvbnN0IGF4aXMgPSAxO1xuICogeC5tZWFuKGF4aXMpLnByaW50KCk7ICAvLyBvciB0Zi5tZWFuKHgsIGF4aXMpXG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0geCBUaGUgaW5wdXQgdGVuc29yLlxuICogQHBhcmFtIGF4aXMgVGhlIGRpbWVuc2lvbihzKSB0byByZWR1Y2UuIEJ5IGRlZmF1bHQgaXQgcmVkdWNlc1xuICogICAgIGFsbCBkaW1lbnNpb25zLlxuICogQHBhcmFtIGtlZXBEaW1zIElmIHRydWUsIHJldGFpbnMgcmVkdWNlZCBkaW1lbnNpb25zIHdpdGggc2l6ZSAxLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ1JlZHVjdGlvbid9XG4gKi9cbmZ1bmN0aW9uIG1lYW5fPFQgZXh0ZW5kcyBUZW5zb3I+KFxuICAgIHg6IFRlbnNvcnxUZW5zb3JMaWtlLCBheGlzOiBudW1iZXJ8bnVtYmVyW10gPSBudWxsLCBrZWVwRGltcyA9IGZhbHNlKTogVCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ21lYW4nKTtcblxuICBjb25zdCBpbnB1dHM6IE1lYW5JbnB1dHMgPSB7eDogJHh9O1xuICBjb25zdCBhdHRyczogTWVhbkF0dHJzID0ge2F4aXMsIGtlZXBEaW1zfTtcblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgIE1lYW4sIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3QgbWVhbiA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7bWVhbl99KTtcbiJdfQ==