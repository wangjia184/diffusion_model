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
import { RealDiv } from '../kernel_names';
import { makeTypesMatch } from '../tensor_util';
import { convertToTensor } from '../tensor_util_env';
import { floorDiv } from './floorDiv';
import { op } from './operation';
/**
 * Divides two `tf.Tensor`s element-wise, A / B. Supports broadcasting.
 *
 * ```js
 * const a = tf.tensor1d([1, 4, 9, 16]);
 * const b = tf.tensor1d([1, 2, 3, 4]);
 *
 * a.div(b).print();  // or tf.div(a, b)
 * ```
 *
 * ```js
 * // Broadcast div a with b.
 * const a = tf.tensor1d([2, 4, 6, 8]);
 * const b = tf.scalar(2);
 *
 * a.div(b).print();  // or tf.div(a, b)
 * ```
 *
 * @param a The first tensor as the numerator.
 * @param b The second tensor as the denominator. Must have the same dtype as
 * `a`.
 *
 * @doc {heading: 'Operations', subheading: 'Arithmetic'}
 */
function div_(a, b) {
    let $a = convertToTensor(a, 'a', 'div');
    let $b = convertToTensor(b, 'b', 'div');
    [$a, $b] = makeTypesMatch($a, $b);
    if ($a.dtype === 'int32' && $b.dtype === 'int32') {
        return floorDiv($a, $b);
    }
    const inputs = { a: $a, b: $b };
    const attrs = {};
    // tslint:disable-next-line: no-unnecessary-type-assertion
    return ENGINE.runKernel(RealDiv, inputs, attrs);
}
export const div = /* @__PURE__ */ op({ div_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZGl2LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZGl2LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLE9BQU8sRUFBZ0IsTUFBTSxpQkFBaUIsQ0FBQztBQUd2RCxPQUFPLEVBQUMsY0FBYyxFQUFDLE1BQU0sZ0JBQWdCLENBQUM7QUFDOUMsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBR25ELE9BQU8sRUFBQyxRQUFRLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDcEMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0F1Qkc7QUFDSCxTQUFTLElBQUksQ0FBbUIsQ0FBb0IsRUFBRSxDQUFvQjtJQUN4RSxJQUFJLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUN4QyxJQUFJLEVBQUUsR0FBRyxlQUFlLENBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUN4QyxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsR0FBRyxjQUFjLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO0lBRWxDLElBQUksRUFBRSxDQUFDLEtBQUssS0FBSyxPQUFPLElBQUksRUFBRSxDQUFDLEtBQUssS0FBSyxPQUFPLEVBQUU7UUFDaEQsT0FBTyxRQUFRLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDO0tBQ3pCO0lBRUQsTUFBTSxNQUFNLEdBQWtCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFDLENBQUM7SUFDN0MsTUFBTSxLQUFLLEdBQUcsRUFBRSxDQUFDO0lBRWpCLDBEQUEwRDtJQUMxRCxPQUFPLE1BQU0sQ0FBQyxTQUFTLENBQUMsT0FBTyxFQUNQLE1BQW1DLEVBQUUsS0FBSyxDQUFNLENBQUM7QUFDM0UsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLEdBQUcsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsSUFBSSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1JlYWxEaXYsIFJlYWxEaXZJbnB1dHN9IGZyb20gJy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge21ha2VUeXBlc01hdGNofSBmcm9tICcuLi90ZW5zb3JfdXRpbCc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge2Zsb29yRGl2fSBmcm9tICcuL2Zsb29yRGl2JztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBEaXZpZGVzIHR3byBgdGYuVGVuc29yYHMgZWxlbWVudC13aXNlLCBBIC8gQi4gU3VwcG9ydHMgYnJvYWRjYXN0aW5nLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBhID0gdGYudGVuc29yMWQoWzEsIDQsIDksIDE2XSk7XG4gKiBjb25zdCBiID0gdGYudGVuc29yMWQoWzEsIDIsIDMsIDRdKTtcbiAqXG4gKiBhLmRpdihiKS5wcmludCgpOyAgLy8gb3IgdGYuZGl2KGEsIGIpXG4gKiBgYGBcbiAqXG4gKiBgYGBqc1xuICogLy8gQnJvYWRjYXN0IGRpdiBhIHdpdGggYi5cbiAqIGNvbnN0IGEgPSB0Zi50ZW5zb3IxZChbMiwgNCwgNiwgOF0pO1xuICogY29uc3QgYiA9IHRmLnNjYWxhcigyKTtcbiAqXG4gKiBhLmRpdihiKS5wcmludCgpOyAgLy8gb3IgdGYuZGl2KGEsIGIpXG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gYSBUaGUgZmlyc3QgdGVuc29yIGFzIHRoZSBudW1lcmF0b3IuXG4gKiBAcGFyYW0gYiBUaGUgc2Vjb25kIHRlbnNvciBhcyB0aGUgZGVub21pbmF0b3IuIE11c3QgaGF2ZSB0aGUgc2FtZSBkdHlwZSBhc1xuICogYGFgLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ0FyaXRobWV0aWMnfVxuICovXG5mdW5jdGlvbiBkaXZfPFQgZXh0ZW5kcyBUZW5zb3I+KGE6IFRlbnNvcnxUZW5zb3JMaWtlLCBiOiBUZW5zb3J8VGVuc29yTGlrZSk6IFQge1xuICBsZXQgJGEgPSBjb252ZXJ0VG9UZW5zb3IoYSwgJ2EnLCAnZGl2Jyk7XG4gIGxldCAkYiA9IGNvbnZlcnRUb1RlbnNvcihiLCAnYicsICdkaXYnKTtcbiAgWyRhLCAkYl0gPSBtYWtlVHlwZXNNYXRjaCgkYSwgJGIpO1xuXG4gIGlmICgkYS5kdHlwZSA9PT0gJ2ludDMyJyAmJiAkYi5kdHlwZSA9PT0gJ2ludDMyJykge1xuICAgIHJldHVybiBmbG9vckRpdigkYSwgJGIpO1xuICB9XG5cbiAgY29uc3QgaW5wdXRzOiBSZWFsRGl2SW5wdXRzID0ge2E6ICRhLCBiOiAkYn07XG4gIGNvbnN0IGF0dHJzID0ge307XG5cbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOiBuby11bm5lY2Vzc2FyeS10eXBlLWFzc2VydGlvblxuICByZXR1cm4gRU5HSU5FLnJ1bktlcm5lbChSZWFsRGl2LFxuICAgICAgICAgICAgICAgICAgICAgICAgICBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCwgYXR0cnMpIGFzIFQ7XG59XG5cbmV4cG9ydCBjb25zdCBkaXYgPSAvKiBAX19QVVJFX18gKi8gb3Aoe2Rpdl99KTtcbiJdfQ==