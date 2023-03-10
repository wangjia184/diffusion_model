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
import { SearchSorted } from '@tensorflow/tfjs-core';
import { searchSortedImpl } from './SearchSorted_impl';
export function searchSorted(args) {
    const { inputs, backend, attrs } = args;
    const { sortedSequence, values } = inputs;
    const { side } = attrs;
    const $sortedSequence = backend.data.get(sortedSequence.dataId).values;
    const $values = backend.data.get(values.dataId).values;
    const output = searchSortedImpl($sortedSequence, $values, sortedSequence.shape[0], sortedSequence.shape[1], values.shape[1], side);
    return backend.makeTensorInfo(values.shape, 'int32', output);
}
export const searchSortedConfig = {
    kernelName: SearchSorted,
    backendName: 'cpu',
    kernelFunc: searchSorted,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU2VhcmNoU29ydGVkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLWNwdS9zcmMva2VybmVscy9TZWFyY2hTb3J0ZWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUEyQixZQUFZLEVBQWdFLE1BQU0sdUJBQXVCLENBQUM7QUFJNUksT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFFckQsTUFBTSxVQUFVLFlBQVksQ0FBQyxJQUk1QjtJQUNDLE1BQU0sRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBQyxHQUFHLElBQUksQ0FBQztJQUN0QyxNQUFNLEVBQUMsY0FBYyxFQUFFLE1BQU0sRUFBQyxHQUFHLE1BQU0sQ0FBQztJQUN4QyxNQUFNLEVBQUMsSUFBSSxFQUFDLEdBQUcsS0FBSyxDQUFDO0lBRXJCLE1BQU0sZUFBZSxHQUNqQixPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxjQUFjLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBb0IsQ0FBQztJQUNqRSxNQUFNLE9BQU8sR0FBRyxPQUFPLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxNQUFNLENBQUMsTUFBTSxDQUFDLENBQUMsTUFBb0IsQ0FBQztJQUVyRSxNQUFNLE1BQU0sR0FBRyxnQkFBZ0IsQ0FDM0IsZUFBZSxFQUFFLE9BQU8sRUFBRSxjQUFjLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUNqRCxjQUFjLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDcEQsT0FBTyxPQUFPLENBQUMsY0FBYyxDQUFDLE1BQU0sQ0FBQyxLQUFLLEVBQUUsT0FBTyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0FBQy9ELENBQUM7QUFFRCxNQUFNLENBQUMsTUFBTSxrQkFBa0IsR0FBaUI7SUFDOUMsVUFBVSxFQUFFLFlBQVk7SUFDeEIsV0FBVyxFQUFFLEtBQUs7SUFDbEIsVUFBVSxFQUFFLFlBQXFDO0NBQ2xELENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMiBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7S2VybmVsQ29uZmlnLCBLZXJuZWxGdW5jLCBTZWFyY2hTb3J0ZWQsIFNlYXJjaFNvcnRlZEF0dHJzLCBTZWFyY2hTb3J0ZWRJbnB1dHMsIFRlbnNvckluZm8sIFR5cGVkQXJyYXl9IGZyb20gJ0B0ZW5zb3JmbG93L3RmanMtY29yZSc7XG5cbmltcG9ydCB7TWF0aEJhY2tlbmRDUFV9IGZyb20gJy4uL2JhY2tlbmRfY3B1JztcblxuaW1wb3J0IHtzZWFyY2hTb3J0ZWRJbXBsfSBmcm9tICcuL1NlYXJjaFNvcnRlZF9pbXBsJztcblxuZXhwb3J0IGZ1bmN0aW9uIHNlYXJjaFNvcnRlZChhcmdzOiB7XG4gIGlucHV0czogU2VhcmNoU29ydGVkSW5wdXRzLFxuICBiYWNrZW5kOiBNYXRoQmFja2VuZENQVSxcbiAgYXR0cnM6IFNlYXJjaFNvcnRlZEF0dHJzXG59KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtzb3J0ZWRTZXF1ZW5jZSwgdmFsdWVzfSA9IGlucHV0cztcbiAgY29uc3Qge3NpZGV9ID0gYXR0cnM7XG5cbiAgY29uc3QgJHNvcnRlZFNlcXVlbmNlID1cbiAgICAgIGJhY2tlbmQuZGF0YS5nZXQoc29ydGVkU2VxdWVuY2UuZGF0YUlkKS52YWx1ZXMgYXMgVHlwZWRBcnJheTtcbiAgY29uc3QgJHZhbHVlcyA9IGJhY2tlbmQuZGF0YS5nZXQodmFsdWVzLmRhdGFJZCkudmFsdWVzIGFzIFR5cGVkQXJyYXk7XG5cbiAgY29uc3Qgb3V0cHV0ID0gc2VhcmNoU29ydGVkSW1wbChcbiAgICAgICRzb3J0ZWRTZXF1ZW5jZSwgJHZhbHVlcywgc29ydGVkU2VxdWVuY2Uuc2hhcGVbMF0sXG4gICAgICBzb3J0ZWRTZXF1ZW5jZS5zaGFwZVsxXSwgdmFsdWVzLnNoYXBlWzFdLCBzaWRlKTtcbiAgcmV0dXJuIGJhY2tlbmQubWFrZVRlbnNvckluZm8odmFsdWVzLnNoYXBlLCAnaW50MzInLCBvdXRwdXQpO1xufVxuXG5leHBvcnQgY29uc3Qgc2VhcmNoU29ydGVkQ29uZmlnOiBLZXJuZWxDb25maWcgPSB7XG4gIGtlcm5lbE5hbWU6IFNlYXJjaFNvcnRlZCxcbiAgYmFja2VuZE5hbWU6ICdjcHUnLFxuICBrZXJuZWxGdW5jOiBzZWFyY2hTb3J0ZWQgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jLFxufTtcbiJdfQ==