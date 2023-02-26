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
import { Slice } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Extracts a slice from a `tf.Tensor` starting at coordinates `begin`
 * and is of size `size`.
 *
 * Also available are stricter rank-specific methods with the same signature
 * as this method that assert that `x` is of the given rank:
 *   - `tf.slice1d`
 *   - `tf.slice2d`
 *   - `tf.slice3d`
 *   - `tf.slice4d`
 *
 * ```js
 * const x = tf.tensor1d([1, 2, 3, 4]);
 *
 * x.slice([1], [2]).print();
 * ```
 *
 * ```js
 * const x = tf.tensor2d([1, 2, 3, 4], [2, 2]);
 *
 * x.slice([1, 0], [1, 2]).print();
 * ```
 * @param x The input `tf.Tensor` to slice from.
 * @param begin The coordinates to start the slice from. The length can be
 *     less than the rank of x - the rest of the axes will have implicit 0 as
 *     start. Can also be a single number, in which case it specifies the
 *     first axis.
 * @param size The size of the slice. The length can be less than the rank of
 *     x - the rest of the axes will have implicit -1. A value of -1 requests
 *     the rest of the dimensions in the axis. Can also be a single number,
 *     in which case it specifies the size of the first axis.
 *
 * @doc {heading: 'Tensors', subheading: 'Slicing and Joining'}
 */
function slice_(x, begin, size) {
    const $x = convertToTensor(x, 'x', 'slice', 'string_or_numeric');
    if ($x.rank === 0) {
        throw new Error('Slicing scalar is not possible');
    }
    const inputs = { x: $x };
    const attrs = { begin, size };
    return ENGINE.runKernel(Slice, inputs, attrs);
}
export const slice = /* @__PURE__ */ op({ slice_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2xpY2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zbGljZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxLQUFLLEVBQTBCLE1BQU0saUJBQWlCLENBQUM7QUFJL0QsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBR25ELE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQWlDRztBQUNILFNBQVMsTUFBTSxDQUNYLENBQWUsRUFBRSxLQUFzQixFQUFFLElBQXNCO0lBQ2pFLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE9BQU8sRUFBRSxtQkFBbUIsQ0FBQyxDQUFDO0lBRWpFLElBQUksRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDakIsTUFBTSxJQUFJLEtBQUssQ0FBQyxnQ0FBZ0MsQ0FBQyxDQUFDO0tBQ25EO0lBRUQsTUFBTSxNQUFNLEdBQWdCLEVBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBQyxDQUFDO0lBQ3BDLE1BQU0sS0FBSyxHQUFlLEVBQUMsS0FBSyxFQUFFLElBQUksRUFBQyxDQUFDO0lBRXhDLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FDbkIsS0FBSyxFQUFFLE1BQW1DLEVBQzFDLEtBQWdDLENBQUMsQ0FBQztBQUN4QyxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sS0FBSyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxNQUFNLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vZW5naW5lJztcbmltcG9ydCB7U2xpY2UsIFNsaWNlQXR0cnMsIFNsaWNlSW5wdXRzfSBmcm9tICcuLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7UmFuaywgVGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogRXh0cmFjdHMgYSBzbGljZSBmcm9tIGEgYHRmLlRlbnNvcmAgc3RhcnRpbmcgYXQgY29vcmRpbmF0ZXMgYGJlZ2luYFxuICogYW5kIGlzIG9mIHNpemUgYHNpemVgLlxuICpcbiAqIEFsc28gYXZhaWxhYmxlIGFyZSBzdHJpY3RlciByYW5rLXNwZWNpZmljIG1ldGhvZHMgd2l0aCB0aGUgc2FtZSBzaWduYXR1cmVcbiAqIGFzIHRoaXMgbWV0aG9kIHRoYXQgYXNzZXJ0IHRoYXQgYHhgIGlzIG9mIHRoZSBnaXZlbiByYW5rOlxuICogICAtIGB0Zi5zbGljZTFkYFxuICogICAtIGB0Zi5zbGljZTJkYFxuICogICAtIGB0Zi5zbGljZTNkYFxuICogICAtIGB0Zi5zbGljZTRkYFxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCB4ID0gdGYudGVuc29yMWQoWzEsIDIsIDMsIDRdKTtcbiAqXG4gKiB4LnNsaWNlKFsxXSwgWzJdKS5wcmludCgpO1xuICogYGBgXG4gKlxuICogYGBganNcbiAqIGNvbnN0IHggPSB0Zi50ZW5zb3IyZChbMSwgMiwgMywgNF0sIFsyLCAyXSk7XG4gKlxuICogeC5zbGljZShbMSwgMF0sIFsxLCAyXSkucHJpbnQoKTtcbiAqIGBgYFxuICogQHBhcmFtIHggVGhlIGlucHV0IGB0Zi5UZW5zb3JgIHRvIHNsaWNlIGZyb20uXG4gKiBAcGFyYW0gYmVnaW4gVGhlIGNvb3JkaW5hdGVzIHRvIHN0YXJ0IHRoZSBzbGljZSBmcm9tLiBUaGUgbGVuZ3RoIGNhbiBiZVxuICogICAgIGxlc3MgdGhhbiB0aGUgcmFuayBvZiB4IC0gdGhlIHJlc3Qgb2YgdGhlIGF4ZXMgd2lsbCBoYXZlIGltcGxpY2l0IDAgYXNcbiAqICAgICBzdGFydC4gQ2FuIGFsc28gYmUgYSBzaW5nbGUgbnVtYmVyLCBpbiB3aGljaCBjYXNlIGl0IHNwZWNpZmllcyB0aGVcbiAqICAgICBmaXJzdCBheGlzLlxuICogQHBhcmFtIHNpemUgVGhlIHNpemUgb2YgdGhlIHNsaWNlLiBUaGUgbGVuZ3RoIGNhbiBiZSBsZXNzIHRoYW4gdGhlIHJhbmsgb2ZcbiAqICAgICB4IC0gdGhlIHJlc3Qgb2YgdGhlIGF4ZXMgd2lsbCBoYXZlIGltcGxpY2l0IC0xLiBBIHZhbHVlIG9mIC0xIHJlcXVlc3RzXG4gKiAgICAgdGhlIHJlc3Qgb2YgdGhlIGRpbWVuc2lvbnMgaW4gdGhlIGF4aXMuIENhbiBhbHNvIGJlIGEgc2luZ2xlIG51bWJlcixcbiAqICAgICBpbiB3aGljaCBjYXNlIGl0IHNwZWNpZmllcyB0aGUgc2l6ZSBvZiB0aGUgZmlyc3QgYXhpcy5cbiAqXG4gKiBAZG9jIHtoZWFkaW5nOiAnVGVuc29ycycsIHN1YmhlYWRpbmc6ICdTbGljaW5nIGFuZCBKb2luaW5nJ31cbiAqL1xuZnVuY3Rpb24gc2xpY2VfPFIgZXh0ZW5kcyBSYW5rLCBUIGV4dGVuZHMgVGVuc29yPFI+PihcbiAgICB4OiBUfFRlbnNvckxpa2UsIGJlZ2luOiBudW1iZXJ8bnVtYmVyW10sIHNpemU/OiBudW1iZXJ8bnVtYmVyW10pOiBUIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnc2xpY2UnLCAnc3RyaW5nX29yX251bWVyaWMnKTtcblxuICBpZiAoJHgucmFuayA9PT0gMCkge1xuICAgIHRocm93IG5ldyBFcnJvcignU2xpY2luZyBzY2FsYXIgaXMgbm90IHBvc3NpYmxlJyk7XG4gIH1cblxuICBjb25zdCBpbnB1dHM6IFNsaWNlSW5wdXRzID0ge3g6ICR4fTtcbiAgY29uc3QgYXR0cnM6IFNsaWNlQXR0cnMgPSB7YmVnaW4sIHNpemV9O1xuXG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgU2xpY2UsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApO1xufVxuXG5leHBvcnQgY29uc3Qgc2xpY2UgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3NsaWNlX30pO1xuIl19