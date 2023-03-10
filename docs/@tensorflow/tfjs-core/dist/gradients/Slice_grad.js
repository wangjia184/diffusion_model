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
import { Slice } from '../kernel_names';
import { pad } from '../ops/pad';
import { parseSliceParams } from '../ops/slice_util';
export const sliceGradConfig = {
    kernelName: Slice,
    inputsToSave: ['x'],
    gradFunc: (dy, saved, attrs) => {
        const [x] = saved;
        const { begin, size } = attrs;
        const inputShape = x.shape;
        const [begin_, size_] = parseSliceParams(x, begin, size);
        // Create an Nx2 padding where the first column represents how many
        // zeros are prepended (at start) for each dimension, and the second
        // column indicates how many zeros are appended (at end).
        // The number of zeros to append is the shape of the input
        // elementwise-subtracted by both the begin vector and sizes vector.
        const paddings = [];
        for (let i = 0; i < dy.rank; i++) {
            paddings.push([begin_[i], inputShape[i] - begin_[i] - size_[i]]);
        }
        return { x: () => pad(dy, paddings) };
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiU2xpY2VfZ3JhZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvZ3JhZGllbnRzL1NsaWNlX2dyYWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxFQUFDLEtBQUssRUFBYSxNQUFNLGlCQUFpQixDQUFDO0FBRWxELE9BQU8sRUFBQyxHQUFHLEVBQUMsTUFBTSxZQUFZLENBQUM7QUFDL0IsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0sbUJBQW1CLENBQUM7QUFHbkQsTUFBTSxDQUFDLE1BQU0sZUFBZSxHQUFlO0lBQ3pDLFVBQVUsRUFBRSxLQUFLO0lBQ2pCLFlBQVksRUFBRSxDQUFDLEdBQUcsQ0FBQztJQUNuQixRQUFRLEVBQUUsQ0FBQyxFQUFVLEVBQUUsS0FBZSxFQUFFLEtBQW1CLEVBQUUsRUFBRTtRQUM3RCxNQUFNLENBQUMsQ0FBQyxDQUFDLEdBQUcsS0FBSyxDQUFDO1FBQ2xCLE1BQU0sRUFBQyxLQUFLLEVBQUUsSUFBSSxFQUFDLEdBQUcsS0FBOEIsQ0FBQztRQUVyRCxNQUFNLFVBQVUsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDO1FBQzNCLE1BQU0sQ0FBQyxNQUFNLEVBQUUsS0FBSyxDQUFDLEdBQUcsZ0JBQWdCLENBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxJQUFJLENBQUMsQ0FBQztRQUV6RCxtRUFBbUU7UUFDbkUsb0VBQW9FO1FBQ3BFLHlEQUF5RDtRQUV6RCwwREFBMEQ7UUFDMUQsb0VBQW9FO1FBQ3BFLE1BQU0sUUFBUSxHQUE0QixFQUFFLENBQUM7UUFDN0MsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxFQUFFLEVBQUU7WUFDaEMsUUFBUSxDQUFDLElBQUksQ0FBQyxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxHQUFHLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7U0FDbEU7UUFDRCxPQUFPLEVBQUMsQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxFQUFFLEVBQUUsUUFBUSxDQUFDLEVBQUMsQ0FBQztJQUN0QyxDQUFDO0NBQ0YsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtTbGljZSwgU2xpY2VBdHRyc30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7R3JhZENvbmZpZywgTmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtwYWR9IGZyb20gJy4uL29wcy9wYWQnO1xuaW1wb3J0IHtwYXJzZVNsaWNlUGFyYW1zfSBmcm9tICcuLi9vcHMvc2xpY2VfdXRpbCc7XG5pbXBvcnQge1RlbnNvcn0gZnJvbSAnLi4vdGVuc29yJztcblxuZXhwb3J0IGNvbnN0IHNsaWNlR3JhZENvbmZpZzogR3JhZENvbmZpZyA9IHtcbiAga2VybmVsTmFtZTogU2xpY2UsXG4gIGlucHV0c1RvU2F2ZTogWyd4J10sXG4gIGdyYWRGdW5jOiAoZHk6IFRlbnNvciwgc2F2ZWQ6IFRlbnNvcltdLCBhdHRyczogTmFtZWRBdHRyTWFwKSA9PiB7XG4gICAgY29uc3QgW3hdID0gc2F2ZWQ7XG4gICAgY29uc3Qge2JlZ2luLCBzaXplfSA9IGF0dHJzIGFzIHVua25vd24gYXMgU2xpY2VBdHRycztcblxuICAgIGNvbnN0IGlucHV0U2hhcGUgPSB4LnNoYXBlO1xuICAgIGNvbnN0IFtiZWdpbl8sIHNpemVfXSA9IHBhcnNlU2xpY2VQYXJhbXMoeCwgYmVnaW4sIHNpemUpO1xuXG4gICAgLy8gQ3JlYXRlIGFuIE54MiBwYWRkaW5nIHdoZXJlIHRoZSBmaXJzdCBjb2x1bW4gcmVwcmVzZW50cyBob3cgbWFueVxuICAgIC8vIHplcm9zIGFyZSBwcmVwZW5kZWQgKGF0IHN0YXJ0KSBmb3IgZWFjaCBkaW1lbnNpb24sIGFuZCB0aGUgc2Vjb25kXG4gICAgLy8gY29sdW1uIGluZGljYXRlcyBob3cgbWFueSB6ZXJvcyBhcmUgYXBwZW5kZWQgKGF0IGVuZCkuXG5cbiAgICAvLyBUaGUgbnVtYmVyIG9mIHplcm9zIHRvIGFwcGVuZCBpcyB0aGUgc2hhcGUgb2YgdGhlIGlucHV0XG4gICAgLy8gZWxlbWVudHdpc2Utc3VidHJhY3RlZCBieSBib3RoIHRoZSBiZWdpbiB2ZWN0b3IgYW5kIHNpemVzIHZlY3Rvci5cbiAgICBjb25zdCBwYWRkaW5nczogQXJyYXk8W251bWJlciwgbnVtYmVyXT4gPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGR5LnJhbms7IGkrKykge1xuICAgICAgcGFkZGluZ3MucHVzaChbYmVnaW5fW2ldLCBpbnB1dFNoYXBlW2ldIC0gYmVnaW5fW2ldIC0gc2l6ZV9baV1dKTtcbiAgICB9XG4gICAgcmV0dXJuIHt4OiAoKSA9PiBwYWQoZHksIHBhZGRpbmdzKX07XG4gIH1cbn07XG4iXX0=