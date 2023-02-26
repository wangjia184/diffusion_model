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
import { Select } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { broadcastTo } from './broadcast_to';
import { assertAndGetBroadcastShape } from './broadcast_util';
import { op } from './operation';
/**
 * Returns the elements, either `a` or `b` depending on the `condition`.
 *
 * If the condition is true, select from `a`, otherwise select from `b`.
 *
 * ```js
 * const cond = tf.tensor1d([false, false, true], 'bool');
 * const a = tf.tensor1d([1 , 2, 3]);
 * const b = tf.tensor1d([-1, -2, -3]);
 *
 * a.where(cond, b).print();
 * ```
 *
 * @param condition The input condition. Must be of dtype bool.
 * @param a If `condition` is rank 1, `a` may have a higher rank but
 *     its first dimension must match the size of `condition`.
 * @param b A tensor with the same dtype as `a` and with shape that is
 *     compatible with `a`.
 * @return A tensor with same dtype as `a` and `b`, and shape that is
 *     broadcastable from `a` and `b`.
 *
 * @doc {heading: 'Operations', subheading: 'Logical'}
 */
function where_(condition, a, b) {
    const $a = convertToTensor(a, 'a', 'where');
    const $b = convertToTensor(b, 'b', 'where');
    const $condition = convertToTensor(condition, 'condition', 'where', 'bool');
    // TODO: move this logic to forward function when the broadcastTo op is
    // implemented in WASM.
    // Find the broadcastable shape for $condition, $a, and $b.
    const broadcastShape = assertAndGetBroadcastShape(assertAndGetBroadcastShape($condition.shape, $a.shape), $b.shape);
    const $broadcastedCondition = broadcastTo($condition, broadcastShape);
    const $broadcastedA = broadcastTo($a, broadcastShape);
    const $broadcastedB = broadcastTo($b, broadcastShape);
    const inputs = {
        condition: $broadcastedCondition,
        t: $broadcastedA,
        e: $broadcastedB
    };
    return ENGINE.runKernel(Select, inputs);
}
export const where = /* @__PURE__ */ op({ where_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoid2hlcmUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy93aGVyZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sV0FBVyxDQUFDO0FBQ2pDLE9BQU8sRUFBQyxNQUFNLEVBQWUsTUFBTSxpQkFBaUIsQ0FBQztBQUdyRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFHbkQsT0FBTyxFQUFDLFdBQVcsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQzNDLE9BQU8sRUFBQywwQkFBMEIsRUFBQyxNQUFNLGtCQUFrQixDQUFDO0FBQzVELE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7R0FzQkc7QUFDSCxTQUFTLE1BQU0sQ0FDWCxTQUE0QixFQUFFLENBQWUsRUFBRSxDQUFlO0lBQ2hFLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQzVDLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQzVDLE1BQU0sVUFBVSxHQUFHLGVBQWUsQ0FBQyxTQUFTLEVBQUUsV0FBVyxFQUFFLE9BQU8sRUFBRSxNQUFNLENBQUMsQ0FBQztJQUM1RSx1RUFBdUU7SUFDdkUsdUJBQXVCO0lBQ3ZCLDJEQUEyRDtJQUMzRCxNQUFNLGNBQWMsR0FBRywwQkFBMEIsQ0FDN0MsMEJBQTBCLENBQUMsVUFBVSxDQUFDLEtBQUssRUFBRSxFQUFFLENBQUMsS0FBSyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQ3RFLE1BQU0scUJBQXFCLEdBQUcsV0FBVyxDQUFDLFVBQVUsRUFBRSxjQUFjLENBQUMsQ0FBQztJQUN0RSxNQUFNLGFBQWEsR0FBRyxXQUFXLENBQUMsRUFBRSxFQUFFLGNBQWMsQ0FBQyxDQUFDO0lBQ3RELE1BQU0sYUFBYSxHQUFHLFdBQVcsQ0FBQyxFQUFFLEVBQUUsY0FBYyxDQUFDLENBQUM7SUFFdEQsTUFBTSxNQUFNLEdBQWlCO1FBQzNCLFNBQVMsRUFBRSxxQkFBcUI7UUFDaEMsQ0FBQyxFQUFFLGFBQWE7UUFDaEIsQ0FBQyxFQUFFLGFBQWE7S0FDakIsQ0FBQztJQUNGLE9BQU8sTUFBTSxDQUFDLFNBQVMsQ0FBQyxNQUFNLEVBQUUsTUFBbUMsQ0FBQyxDQUFDO0FBQ3ZFLENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxLQUFLLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLE1BQU0sRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi9lbmdpbmUnO1xuaW1wb3J0IHtTZWxlY3QsIFNlbGVjdElucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5cbmltcG9ydCB7YnJvYWRjYXN0VG99IGZyb20gJy4vYnJvYWRjYXN0X3RvJztcbmltcG9ydCB7YXNzZXJ0QW5kR2V0QnJvYWRjYXN0U2hhcGV9IGZyb20gJy4vYnJvYWRjYXN0X3V0aWwnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIFJldHVybnMgdGhlIGVsZW1lbnRzLCBlaXRoZXIgYGFgIG9yIGBiYCBkZXBlbmRpbmcgb24gdGhlIGBjb25kaXRpb25gLlxuICpcbiAqIElmIHRoZSBjb25kaXRpb24gaXMgdHJ1ZSwgc2VsZWN0IGZyb20gYGFgLCBvdGhlcndpc2Ugc2VsZWN0IGZyb20gYGJgLlxuICpcbiAqIGBgYGpzXG4gKiBjb25zdCBjb25kID0gdGYudGVuc29yMWQoW2ZhbHNlLCBmYWxzZSwgdHJ1ZV0sICdib29sJyk7XG4gKiBjb25zdCBhID0gdGYudGVuc29yMWQoWzEgLCAyLCAzXSk7XG4gKiBjb25zdCBiID0gdGYudGVuc29yMWQoWy0xLCAtMiwgLTNdKTtcbiAqXG4gKiBhLndoZXJlKGNvbmQsIGIpLnByaW50KCk7XG4gKiBgYGBcbiAqXG4gKiBAcGFyYW0gY29uZGl0aW9uIFRoZSBpbnB1dCBjb25kaXRpb24uIE11c3QgYmUgb2YgZHR5cGUgYm9vbC5cbiAqIEBwYXJhbSBhIElmIGBjb25kaXRpb25gIGlzIHJhbmsgMSwgYGFgIG1heSBoYXZlIGEgaGlnaGVyIHJhbmsgYnV0XG4gKiAgICAgaXRzIGZpcnN0IGRpbWVuc2lvbiBtdXN0IG1hdGNoIHRoZSBzaXplIG9mIGBjb25kaXRpb25gLlxuICogQHBhcmFtIGIgQSB0ZW5zb3Igd2l0aCB0aGUgc2FtZSBkdHlwZSBhcyBgYWAgYW5kIHdpdGggc2hhcGUgdGhhdCBpc1xuICogICAgIGNvbXBhdGlibGUgd2l0aCBgYWAuXG4gKiBAcmV0dXJuIEEgdGVuc29yIHdpdGggc2FtZSBkdHlwZSBhcyBgYWAgYW5kIGBiYCwgYW5kIHNoYXBlIHRoYXQgaXNcbiAqICAgICBicm9hZGNhc3RhYmxlIGZyb20gYGFgIGFuZCBgYmAuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnTG9naWNhbCd9XG4gKi9cbmZ1bmN0aW9uIHdoZXJlXzxUIGV4dGVuZHMgVGVuc29yPihcbiAgICBjb25kaXRpb246IFRlbnNvcnxUZW5zb3JMaWtlLCBhOiBUfFRlbnNvckxpa2UsIGI6IFR8VGVuc29yTGlrZSk6IFQge1xuICBjb25zdCAkYSA9IGNvbnZlcnRUb1RlbnNvcihhLCAnYScsICd3aGVyZScpO1xuICBjb25zdCAkYiA9IGNvbnZlcnRUb1RlbnNvcihiLCAnYicsICd3aGVyZScpO1xuICBjb25zdCAkY29uZGl0aW9uID0gY29udmVydFRvVGVuc29yKGNvbmRpdGlvbiwgJ2NvbmRpdGlvbicsICd3aGVyZScsICdib29sJyk7XG4gIC8vIFRPRE86IG1vdmUgdGhpcyBsb2dpYyB0byBmb3J3YXJkIGZ1bmN0aW9uIHdoZW4gdGhlIGJyb2FkY2FzdFRvIG9wIGlzXG4gIC8vIGltcGxlbWVudGVkIGluIFdBU00uXG4gIC8vIEZpbmQgdGhlIGJyb2FkY2FzdGFibGUgc2hhcGUgZm9yICRjb25kaXRpb24sICRhLCBhbmQgJGIuXG4gIGNvbnN0IGJyb2FkY2FzdFNoYXBlID0gYXNzZXJ0QW5kR2V0QnJvYWRjYXN0U2hhcGUoXG4gICAgICBhc3NlcnRBbmRHZXRCcm9hZGNhc3RTaGFwZSgkY29uZGl0aW9uLnNoYXBlLCAkYS5zaGFwZSksICRiLnNoYXBlKTtcbiAgY29uc3QgJGJyb2FkY2FzdGVkQ29uZGl0aW9uID0gYnJvYWRjYXN0VG8oJGNvbmRpdGlvbiwgYnJvYWRjYXN0U2hhcGUpO1xuICBjb25zdCAkYnJvYWRjYXN0ZWRBID0gYnJvYWRjYXN0VG8oJGEsIGJyb2FkY2FzdFNoYXBlKTtcbiAgY29uc3QgJGJyb2FkY2FzdGVkQiA9IGJyb2FkY2FzdFRvKCRiLCBicm9hZGNhc3RTaGFwZSk7XG5cbiAgY29uc3QgaW5wdXRzOiBTZWxlY3RJbnB1dHMgPSB7XG4gICAgY29uZGl0aW9uOiAkYnJvYWRjYXN0ZWRDb25kaXRpb24sXG4gICAgdDogJGJyb2FkY2FzdGVkQSxcbiAgICBlOiAkYnJvYWRjYXN0ZWRCXG4gIH07XG4gIHJldHVybiBFTkdJTkUucnVuS2VybmVsKFNlbGVjdCwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXApO1xufVxuXG5leHBvcnQgY29uc3Qgd2hlcmUgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3doZXJlX30pO1xuIl19