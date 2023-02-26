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
import { decodeString, encodeString } from '../util';
// Utilities needed by backend consumers of tf-core.
export * from '../ops/axis_util';
export * from '../ops/broadcast_util';
export * from '../ops/concat_util';
export * from '../ops/conv_util';
export * from '../ops/fused_util';
export * from '../ops/fused_types';
export * from '../ops/ragged_to_dense_util';
export * from '../ops/reduce_util';
import * as slice_util from '../ops/slice_util';
export { slice_util };
export { upcastType } from '../types';
export * from '../ops/rotate_util';
export * from '../ops/array_ops_util';
export * from '../ops/gather_nd_util';
export * from '../ops/scatter_nd_util';
export * from '../ops/selu_util';
export * from '../ops/fused_util';
export * from '../ops/erf_util';
export * from '../log';
export * from '../backends/complex_util';
export * from '../backends/einsum_util';
export * from '../ops/split_util';
export * from '../ops/sparse/sparse_fill_empty_rows_util';
export * from '../ops/sparse/sparse_reshape_util';
export * from '../ops/sparse/sparse_segment_reduction_util';
import * as segment_util from '../ops/segment_util';
export { segment_util };
export function fromUint8ToStringArray(vals) {
    try {
        // Decode the bytes into string.
        return vals.map(val => decodeString(val));
    }
    catch (err) {
        throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${err}`);
    }
}
export function fromStringArrayToUint8(strings) {
    return strings.map(s => encodeString(s));
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYmFja2VuZF91dGlsLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9iYWNrZW5kcy9iYWNrZW5kX3V0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLFlBQVksRUFBRSxZQUFZLEVBQUMsTUFBTSxTQUFTLENBQUM7QUFFbkQsb0RBQW9EO0FBQ3BELGNBQWMsa0JBQWtCLENBQUM7QUFDakMsY0FBYyx1QkFBdUIsQ0FBQztBQUN0QyxjQUFjLG9CQUFvQixDQUFDO0FBQ25DLGNBQWMsa0JBQWtCLENBQUM7QUFDakMsY0FBYyxtQkFBbUIsQ0FBQztBQUNsQyxjQUFjLG9CQUFvQixDQUFDO0FBQ25DLGNBQWMsNkJBQTZCLENBQUM7QUFDNUMsY0FBYyxvQkFBb0IsQ0FBQztBQUVuQyxPQUFPLEtBQUssVUFBVSxNQUFNLG1CQUFtQixDQUFDO0FBQ2hELE9BQU8sRUFBQyxVQUFVLEVBQUMsQ0FBQztBQUVwQixPQUFPLEVBQTRCLFVBQVUsRUFBWSxNQUFNLFVBQVUsQ0FBQztBQUUxRSxjQUFjLG9CQUFvQixDQUFDO0FBQ25DLGNBQWMsdUJBQXVCLENBQUM7QUFDdEMsY0FBYyx1QkFBdUIsQ0FBQztBQUN0QyxjQUFjLHdCQUF3QixDQUFDO0FBQ3ZDLGNBQWMsa0JBQWtCLENBQUM7QUFDakMsY0FBYyxtQkFBbUIsQ0FBQztBQUNsQyxjQUFjLGlCQUFpQixDQUFDO0FBQ2hDLGNBQWMsUUFBUSxDQUFDO0FBQ3ZCLGNBQWMsMEJBQTBCLENBQUM7QUFDekMsY0FBYyx5QkFBeUIsQ0FBQztBQUN4QyxjQUFjLG1CQUFtQixDQUFDO0FBQ2xDLGNBQWMsMkNBQTJDLENBQUM7QUFDMUQsY0FBYyxtQ0FBbUMsQ0FBQztBQUNsRCxjQUFjLDZDQUE2QyxDQUFDO0FBRTVELE9BQU8sS0FBSyxZQUFZLE1BQU0scUJBQXFCLENBQUM7QUFDcEQsT0FBTyxFQUFDLFlBQVksRUFBQyxDQUFDO0FBRXRCLE1BQU0sVUFBVSxzQkFBc0IsQ0FBQyxJQUFrQjtJQUN2RCxJQUFJO1FBQ0YsZ0NBQWdDO1FBQ2hDLE9BQU8sSUFBSSxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLFlBQVksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0tBQzNDO0lBQUMsT0FBTyxHQUFHLEVBQUU7UUFDWixNQUFNLElBQUksS0FBSyxDQUNYLDREQUE0RCxHQUFHLEVBQUUsQ0FBQyxDQUFDO0tBQ3hFO0FBQ0gsQ0FBQztBQUVELE1BQU0sVUFBVSxzQkFBc0IsQ0FBQyxPQUFpQjtJQUN0RCxPQUFPLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUMzQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge2RlY29kZVN0cmluZywgZW5jb2RlU3RyaW5nfSBmcm9tICcuLi91dGlsJztcblxuLy8gVXRpbGl0aWVzIG5lZWRlZCBieSBiYWNrZW5kIGNvbnN1bWVycyBvZiB0Zi1jb3JlLlxuZXhwb3J0ICogZnJvbSAnLi4vb3BzL2F4aXNfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvYnJvYWRjYXN0X3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL2NvbmNhdF91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9jb252X3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL2Z1c2VkX3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL2Z1c2VkX3R5cGVzJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9yYWdnZWRfdG9fZGVuc2VfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvcmVkdWNlX3V0aWwnO1xuXG5pbXBvcnQgKiBhcyBzbGljZV91dGlsIGZyb20gJy4uL29wcy9zbGljZV91dGlsJztcbmV4cG9ydCB7c2xpY2VfdXRpbH07XG5cbmV4cG9ydCB7QmFja2VuZFZhbHVlcywgVHlwZWRBcnJheSwgdXBjYXN0VHlwZSwgUGl4ZWxEYXRhfSBmcm9tICcuLi90eXBlcyc7XG5leHBvcnQge01lbW9yeUluZm8sIFRpbWluZ0luZm99IGZyb20gJy4uL2VuZ2luZSc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvcm90YXRlX3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL2FycmF5X29wc191dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9nYXRoZXJfbmRfdXRpbCc7XG5leHBvcnQgKiBmcm9tICcuLi9vcHMvc2NhdHRlcl9uZF91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9zZWx1X3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL2Z1c2VkX3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL2VyZl91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL2xvZyc7XG5leHBvcnQgKiBmcm9tICcuLi9iYWNrZW5kcy9jb21wbGV4X3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vYmFja2VuZHMvZWluc3VtX3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL3NwbGl0X3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL3NwYXJzZS9zcGFyc2VfZmlsbF9lbXB0eV9yb3dzX3V0aWwnO1xuZXhwb3J0ICogZnJvbSAnLi4vb3BzL3NwYXJzZS9zcGFyc2VfcmVzaGFwZV91dGlsJztcbmV4cG9ydCAqIGZyb20gJy4uL29wcy9zcGFyc2Uvc3BhcnNlX3NlZ21lbnRfcmVkdWN0aW9uX3V0aWwnO1xuXG5pbXBvcnQgKiBhcyBzZWdtZW50X3V0aWwgZnJvbSAnLi4vb3BzL3NlZ21lbnRfdXRpbCc7XG5leHBvcnQge3NlZ21lbnRfdXRpbH07XG5cbmV4cG9ydCBmdW5jdGlvbiBmcm9tVWludDhUb1N0cmluZ0FycmF5KHZhbHM6IFVpbnQ4QXJyYXlbXSkge1xuICB0cnkge1xuICAgIC8vIERlY29kZSB0aGUgYnl0ZXMgaW50byBzdHJpbmcuXG4gICAgcmV0dXJuIHZhbHMubWFwKHZhbCA9PiBkZWNvZGVTdHJpbmcodmFsKSk7XG4gIH0gY2F0Y2ggKGVycikge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYEZhaWxlZCB0byBkZWNvZGUgZW5jb2RlZCBzdHJpbmcgYnl0ZXMgaW50byB1dGYtOCwgZXJyb3I6ICR7ZXJyfWApO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBmcm9tU3RyaW5nQXJyYXlUb1VpbnQ4KHN0cmluZ3M6IHN0cmluZ1tdKSB7XG4gIHJldHVybiBzdHJpbmdzLm1hcChzID0+IGVuY29kZVN0cmluZyhzKSk7XG59XG4iXX0=