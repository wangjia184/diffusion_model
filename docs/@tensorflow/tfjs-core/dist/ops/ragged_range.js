/**
 * @license
 * Copyright 2022 Google LLC.
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
import { RaggedRange } from '../kernel_names';
import { convertToTensor } from '../tensor_util_env';
import { op } from './operation';
/**
 * Returns a RaggedTensor result composed from rtDenseValues and rtNestedSplits,
 * such that result[i] = [starts[i], starts[i] + deltas[i], ..., limits[i]]).
 *
 * @param starts: A Tensor. Must be one of the following types:
 *     'float32', 'int32'. The starts of each range.
 * @param limits: A Tensor. Must have the same type as starts. The limits of
 *     each range.
 * @param deltas: A Tensor. Must have the same type as starts. The deltas of
 *     each range.
 * @return A map with the following properties:
 *     - rtNestedSplits: A Tensor of type 'int32'.
 *     - rtDenseValues: A Tensor. Has the same type as starts.
 */
function raggedRange_(starts, limits, deltas) {
    const $starts = convertToTensor(starts, 'starts', 'raggedRange');
    const $limits = convertToTensor(limits, 'limits', 'raggedRange', $starts.dtype);
    const $deltas = convertToTensor(deltas, 'deltas', 'raggedRange', $starts.dtype);
    const inputs = {
        starts: $starts,
        limits: $limits,
        deltas: $deltas,
    };
    const result = ENGINE.runKernel(RaggedRange, inputs);
    return {
        rtNestedSplits: result[0],
        rtDenseValues: result[1],
    };
}
export const raggedRange = /* @__PURE__ */ op({ raggedRange_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmFnZ2VkX3JhbmdlLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvcmFnZ2VkX3JhbmdlLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDakMsT0FBTyxFQUFDLFdBQVcsRUFBb0IsTUFBTSxpQkFBaUIsQ0FBQztBQUcvRCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFFbkQsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGFBQWEsQ0FBQztBQUUvQjs7Ozs7Ozs7Ozs7OztHQWFHO0FBRUgsU0FBUyxZQUFZLENBQ2pCLE1BQXlCLEVBQUUsTUFBeUIsRUFDcEQsTUFBeUI7SUFDM0IsTUFBTSxPQUFPLEdBQUcsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsYUFBYSxDQUFDLENBQUM7SUFDakUsTUFBTSxPQUFPLEdBQ1QsZUFBZSxDQUFDLE1BQU0sRUFBRSxRQUFRLEVBQUUsYUFBYSxFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUNwRSxNQUFNLE9BQU8sR0FDVCxlQUFlLENBQUMsTUFBTSxFQUFFLFFBQVEsRUFBRSxhQUFhLEVBQUUsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBRXBFLE1BQU0sTUFBTSxHQUFzQjtRQUNoQyxNQUFNLEVBQUUsT0FBTztRQUNmLE1BQU0sRUFBRSxPQUFPO1FBQ2YsTUFBTSxFQUFFLE9BQU87S0FDaEIsQ0FBQztJQUVGLE1BQU0sTUFBTSxHQUFhLE1BQU0sQ0FBQyxTQUFTLENBQUMsV0FBVyxFQUFFLE1BQVksQ0FBQyxDQUFDO0lBQ3JFLE9BQU87UUFDTCxjQUFjLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQztRQUN6QixhQUFhLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQztLQUN6QixDQUFDO0FBQ0osQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFdBQVcsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsWUFBWSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uL2VuZ2luZSc7XG5pbXBvcnQge1JhZ2dlZFJhbmdlLCBSYWdnZWRSYW5nZUlucHV0c30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7VGVuc29yfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5cbi8qKlxuICogUmV0dXJucyBhIFJhZ2dlZFRlbnNvciByZXN1bHQgY29tcG9zZWQgZnJvbSBydERlbnNlVmFsdWVzIGFuZCBydE5lc3RlZFNwbGl0cyxcbiAqIHN1Y2ggdGhhdCByZXN1bHRbaV0gPSBbc3RhcnRzW2ldLCBzdGFydHNbaV0gKyBkZWx0YXNbaV0sIC4uLiwgbGltaXRzW2ldXSkuXG4gKlxuICogQHBhcmFtIHN0YXJ0czogQSBUZW5zb3IuIE11c3QgYmUgb25lIG9mIHRoZSBmb2xsb3dpbmcgdHlwZXM6XG4gKiAgICAgJ2Zsb2F0MzInLCAnaW50MzInLiBUaGUgc3RhcnRzIG9mIGVhY2ggcmFuZ2UuXG4gKiBAcGFyYW0gbGltaXRzOiBBIFRlbnNvci4gTXVzdCBoYXZlIHRoZSBzYW1lIHR5cGUgYXMgc3RhcnRzLiBUaGUgbGltaXRzIG9mXG4gKiAgICAgZWFjaCByYW5nZS5cbiAqIEBwYXJhbSBkZWx0YXM6IEEgVGVuc29yLiBNdXN0IGhhdmUgdGhlIHNhbWUgdHlwZSBhcyBzdGFydHMuIFRoZSBkZWx0YXMgb2ZcbiAqICAgICBlYWNoIHJhbmdlLlxuICogQHJldHVybiBBIG1hcCB3aXRoIHRoZSBmb2xsb3dpbmcgcHJvcGVydGllczpcbiAqICAgICAtIHJ0TmVzdGVkU3BsaXRzOiBBIFRlbnNvciBvZiB0eXBlICdpbnQzMicuXG4gKiAgICAgLSBydERlbnNlVmFsdWVzOiBBIFRlbnNvci4gSGFzIHRoZSBzYW1lIHR5cGUgYXMgc3RhcnRzLlxuICovXG5cbmZ1bmN0aW9uIHJhZ2dlZFJhbmdlXyhcbiAgICBzdGFydHM6IFRlbnNvcnxUZW5zb3JMaWtlLCBsaW1pdHM6IFRlbnNvcnxUZW5zb3JMaWtlLFxuICAgIGRlbHRhczogVGVuc29yfFRlbnNvckxpa2UpOiBOYW1lZFRlbnNvck1hcCB7XG4gIGNvbnN0ICRzdGFydHMgPSBjb252ZXJ0VG9UZW5zb3Ioc3RhcnRzLCAnc3RhcnRzJywgJ3JhZ2dlZFJhbmdlJyk7XG4gIGNvbnN0ICRsaW1pdHMgPVxuICAgICAgY29udmVydFRvVGVuc29yKGxpbWl0cywgJ2xpbWl0cycsICdyYWdnZWRSYW5nZScsICRzdGFydHMuZHR5cGUpO1xuICBjb25zdCAkZGVsdGFzID1cbiAgICAgIGNvbnZlcnRUb1RlbnNvcihkZWx0YXMsICdkZWx0YXMnLCAncmFnZ2VkUmFuZ2UnLCAkc3RhcnRzLmR0eXBlKTtcblxuICBjb25zdCBpbnB1dHM6IFJhZ2dlZFJhbmdlSW5wdXRzID0ge1xuICAgIHN0YXJ0czogJHN0YXJ0cyxcbiAgICBsaW1pdHM6ICRsaW1pdHMsXG4gICAgZGVsdGFzOiAkZGVsdGFzLFxuICB9O1xuXG4gIGNvbnN0IHJlc3VsdDogVGVuc29yW10gPSBFTkdJTkUucnVuS2VybmVsKFJhZ2dlZFJhbmdlLCBpbnB1dHMgYXMge30pO1xuICByZXR1cm4ge1xuICAgIHJ0TmVzdGVkU3BsaXRzOiByZXN1bHRbMF0sXG4gICAgcnREZW5zZVZhbHVlczogcmVzdWx0WzFdLFxuICB9O1xufVxuXG5leHBvcnQgY29uc3QgcmFnZ2VkUmFuZ2UgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3JhZ2dlZFJhbmdlX30pO1xuIl19