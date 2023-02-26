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
import { Range } from '../kernel_names';
/**
 * Creates a new `tf.Tensor1D` filled with the numbers in the range provided.
 *
 * The tensor is a half-open interval meaning it includes start, but
 * excludes stop. Decrementing ranges and negative step values are also
 * supported.
 *
 *
 * ```js
 * tf.range(0, 9, 2).print();
 * ```
 *
 * @param start An integer start value
 * @param stop An integer stop value
 * @param step An integer increment (will default to 1 or -1)
 * @param dtype The data type of the output tensor. Defaults to 'float32'.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
export function range(start, stop, step = 1, dtype = 'float32') {
    if (step === 0) {
        throw new Error('Cannot have a step of zero');
    }
    const attrs = { start, stop, step, dtype };
    return ENGINE.runKernel(Range, {} /* inputs */, attrs);
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFuZ2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9yYW5nZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxLQUFLLEVBQWEsTUFBTSxpQkFBaUIsQ0FBQztBQUlsRDs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBa0JHO0FBQ0gsTUFBTSxVQUFVLEtBQUssQ0FDakIsS0FBYSxFQUFFLElBQVksRUFBRSxJQUFJLEdBQUcsQ0FBQyxFQUNyQyxRQUEyQixTQUFTO0lBQ3RDLElBQUksSUFBSSxLQUFLLENBQUMsRUFBRTtRQUNkLE1BQU0sSUFBSSxLQUFLLENBQUMsNEJBQTRCLENBQUMsQ0FBQztLQUMvQztJQUVELE1BQU0sS0FBSyxHQUFlLEVBQUMsS0FBSyxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsS0FBSyxFQUFDLENBQUM7SUFFckQsT0FBTyxNQUFNLENBQUMsU0FBUyxDQUFDLEtBQUssRUFBRSxFQUFFLENBQUMsWUFBWSxFQUMxQyxLQUFnQyxDQUFDLENBQUM7QUFDeEMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1JhbmdlLCBSYW5nZUF0dHJzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjFEfSBmcm9tICcuLi90ZW5zb3InO1xuXG4vKipcbiAqIENyZWF0ZXMgYSBuZXcgYHRmLlRlbnNvcjFEYCBmaWxsZWQgd2l0aCB0aGUgbnVtYmVycyBpbiB0aGUgcmFuZ2UgcHJvdmlkZWQuXG4gKlxuICogVGhlIHRlbnNvciBpcyBhIGhhbGYtb3BlbiBpbnRlcnZhbCBtZWFuaW5nIGl0IGluY2x1ZGVzIHN0YXJ0LCBidXRcbiAqIGV4Y2x1ZGVzIHN0b3AuIERlY3JlbWVudGluZyByYW5nZXMgYW5kIG5lZ2F0aXZlIHN0ZXAgdmFsdWVzIGFyZSBhbHNvXG4gKiBzdXBwb3J0ZWQuXG4gKlxuICpcbiAqIGBgYGpzXG4gKiB0Zi5yYW5nZSgwLCA5LCAyKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogQHBhcmFtIHN0YXJ0IEFuIGludGVnZXIgc3RhcnQgdmFsdWVcbiAqIEBwYXJhbSBzdG9wIEFuIGludGVnZXIgc3RvcCB2YWx1ZVxuICogQHBhcmFtIHN0ZXAgQW4gaW50ZWdlciBpbmNyZW1lbnQgKHdpbGwgZGVmYXVsdCB0byAxIG9yIC0xKVxuICogQHBhcmFtIGR0eXBlIFRoZSBkYXRhIHR5cGUgb2YgdGhlIG91dHB1dCB0ZW5zb3IuIERlZmF1bHRzIHRvICdmbG9hdDMyJy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdDcmVhdGlvbid9XG4gKi9cbmV4cG9ydCBmdW5jdGlvbiByYW5nZShcbiAgICBzdGFydDogbnVtYmVyLCBzdG9wOiBudW1iZXIsIHN0ZXAgPSAxLFxuICAgIGR0eXBlOiAnZmxvYXQzMid8J2ludDMyJyA9ICdmbG9hdDMyJyk6IFRlbnNvcjFEIHtcbiAgaWYgKHN0ZXAgPT09IDApIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0Nhbm5vdCBoYXZlIGEgc3RlcCBvZiB6ZXJvJyk7XG4gIH1cblxuICBjb25zdCBhdHRyczogUmFuZ2VBdHRycyA9IHtzdGFydCwgc3RvcCwgc3RlcCwgZHR5cGV9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFJhbmdlLCB7fSAvKiBpbnB1dHMgKi8sXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG59XG4iXX0=