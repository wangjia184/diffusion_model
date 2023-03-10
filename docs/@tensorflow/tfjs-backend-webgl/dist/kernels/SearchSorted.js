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
import { SearchSortedProgram } from '../search_sorted_gpu';
export function searchSorted(args) {
    const { inputs, backend, attrs } = args;
    const { sortedSequence, values } = inputs;
    const { side } = attrs;
    const program = new SearchSortedProgram(sortedSequence.shape[0], sortedSequence.shape[1], values.shape[1], side);
    const customValues = [[sortedSequence.shape[1]]];
    return backend.runWebGLProgram(program, [sortedSequence, values], 'int32', customValues);
}
export const searchSortedConfig = {
    kernelName: SearchSorted,
    backendName: 'webgl',
    kernelFunc: searchSorted,
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU2VhcmNoU29ydGVkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1iYWNrZW5kLXdlYmdsL3NyYy9rZXJuZWxzL1NlYXJjaFNvcnRlZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQTJCLFlBQVksRUFBb0QsTUFBTSx1QkFBdUIsQ0FBQztBQUdoSSxPQUFPLEVBQUMsbUJBQW1CLEVBQUMsTUFBTSxzQkFBc0IsQ0FBQztBQUV6RCxNQUFNLFVBQVUsWUFBWSxDQUFDLElBSTVCO0lBQ0MsTUFBTSxFQUFDLE1BQU0sRUFBRSxPQUFPLEVBQUUsS0FBSyxFQUFDLEdBQUcsSUFBSSxDQUFDO0lBQ3RDLE1BQU0sRUFBQyxjQUFjLEVBQUUsTUFBTSxFQUFDLEdBQUcsTUFBTSxDQUFDO0lBQ3hDLE1BQU0sRUFBQyxJQUFJLEVBQUMsR0FBRyxLQUFLLENBQUM7SUFFckIsTUFBTSxPQUFPLEdBQUcsSUFBSSxtQkFBbUIsQ0FDbkMsY0FBYyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxjQUFjLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDN0UsTUFBTSxZQUFZLEdBQUcsQ0FBQyxDQUFDLGNBQWMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2pELE9BQU8sT0FBTyxDQUFDLGVBQWUsQ0FDMUIsT0FBTyxFQUFFLENBQUMsY0FBYyxFQUFFLE1BQU0sQ0FBQyxFQUFFLE9BQU8sRUFBRSxZQUFZLENBQUMsQ0FBQztBQUNoRSxDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sa0JBQWtCLEdBQWlCO0lBQzlDLFVBQVUsRUFBRSxZQUFZO0lBQ3hCLFdBQVcsRUFBRSxPQUFPO0lBQ3BCLFVBQVUsRUFBRSxZQUFxQztDQUNsRCxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjIgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0tlcm5lbENvbmZpZywgS2VybmVsRnVuYywgU2VhcmNoU29ydGVkLCBTZWFyY2hTb3J0ZWRBdHRycywgU2VhcmNoU29ydGVkSW5wdXRzLCBUZW5zb3JJbmZvfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuXG5pbXBvcnQge01hdGhCYWNrZW5kV2ViR0x9IGZyb20gJy4uL2JhY2tlbmRfd2ViZ2wnO1xuaW1wb3J0IHtTZWFyY2hTb3J0ZWRQcm9ncmFtfSBmcm9tICcuLi9zZWFyY2hfc29ydGVkX2dwdSc7XG5cbmV4cG9ydCBmdW5jdGlvbiBzZWFyY2hTb3J0ZWQoYXJnczoge1xuICBpbnB1dHM6IFNlYXJjaFNvcnRlZElucHV0cyxcbiAgYmFja2VuZDogTWF0aEJhY2tlbmRXZWJHTCxcbiAgYXR0cnM6IFNlYXJjaFNvcnRlZEF0dHJzXG59KTogVGVuc29ySW5mbyB7XG4gIGNvbnN0IHtpbnB1dHMsIGJhY2tlbmQsIGF0dHJzfSA9IGFyZ3M7XG4gIGNvbnN0IHtzb3J0ZWRTZXF1ZW5jZSwgdmFsdWVzfSA9IGlucHV0cztcbiAgY29uc3Qge3NpZGV9ID0gYXR0cnM7XG5cbiAgY29uc3QgcHJvZ3JhbSA9IG5ldyBTZWFyY2hTb3J0ZWRQcm9ncmFtKFxuICAgICAgc29ydGVkU2VxdWVuY2Uuc2hhcGVbMF0sIHNvcnRlZFNlcXVlbmNlLnNoYXBlWzFdLCB2YWx1ZXMuc2hhcGVbMV0sIHNpZGUpO1xuICBjb25zdCBjdXN0b21WYWx1ZXMgPSBbW3NvcnRlZFNlcXVlbmNlLnNoYXBlWzFdXV07XG4gIHJldHVybiBiYWNrZW5kLnJ1bldlYkdMUHJvZ3JhbShcbiAgICAgIHByb2dyYW0sIFtzb3J0ZWRTZXF1ZW5jZSwgdmFsdWVzXSwgJ2ludDMyJywgY3VzdG9tVmFsdWVzKTtcbn1cblxuZXhwb3J0IGNvbnN0IHNlYXJjaFNvcnRlZENvbmZpZzogS2VybmVsQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBTZWFyY2hTb3J0ZWQsXG4gIGJhY2tlbmROYW1lOiAnd2ViZ2wnLFxuICBrZXJuZWxGdW5jOiBzZWFyY2hTb3J0ZWQgYXMgdW5rbm93biBhcyBLZXJuZWxGdW5jLFxufTtcbiJdfQ==