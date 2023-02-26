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
import { Conv2D } from '../kernel_names';
import { conv2DBackpropFilter } from '../ops/conv2d_backprop_filter';
import { conv2DBackpropInput } from '../ops/conv2d_backprop_input';
import * as conv_util from '../ops/conv_util';
import * as util from '../util';
export const conv2DGradConfig = {
    kernelName: Conv2D,
    inputsToSave: ['x', 'filter'],
    gradFunc: (dy, saved, attrs) => {
        const [x4D, $filter] = saved;
        const { dilations, strides, pad, dataFormat } = attrs;
        util.assert(conv_util.tupleValuesAreOne(dilations), () => 'Error in gradient of conv2D: dilation rates greater than 1 ' +
            `are not yet supported in gradients. Got dilations '${dilations}'`);
        return {
            x: () => conv2DBackpropInput(x4D.shape, dy, $filter, strides, pad, dataFormat),
            filter: () => conv2DBackpropFilter(x4D, dy, $filter.shape, strides, pad, dataFormat)
        };
    }
};
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiQ29udjJEX2dyYWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL2dyYWRpZW50cy9Db252MkRfZ3JhZC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFDSCxPQUFPLEVBQUMsTUFBTSxFQUFjLE1BQU0saUJBQWlCLENBQUM7QUFFcEQsT0FBTyxFQUFDLG9CQUFvQixFQUFDLE1BQU0sK0JBQStCLENBQUM7QUFDbkUsT0FBTyxFQUFDLG1CQUFtQixFQUFDLE1BQU0sOEJBQThCLENBQUM7QUFDakUsT0FBTyxLQUFLLFNBQVMsTUFBTSxrQkFBa0IsQ0FBQztBQUU5QyxPQUFPLEtBQUssSUFBSSxNQUFNLFNBQVMsQ0FBQztBQUVoQyxNQUFNLENBQUMsTUFBTSxnQkFBZ0IsR0FBZTtJQUMxQyxVQUFVLEVBQUUsTUFBTTtJQUNsQixZQUFZLEVBQUUsQ0FBQyxHQUFHLEVBQUUsUUFBUSxDQUFDO0lBQzdCLFFBQVEsRUFBRSxDQUFDLEVBQVksRUFBRSxLQUFlLEVBQUUsS0FBbUIsRUFBRSxFQUFFO1FBQy9ELE1BQU0sQ0FBQyxHQUFHLEVBQUUsT0FBTyxDQUFDLEdBQUcsS0FBNkIsQ0FBQztRQUNyRCxNQUFNLEVBQUMsU0FBUyxFQUFFLE9BQU8sRUFBRSxHQUFHLEVBQUUsVUFBVSxFQUFDLEdBQ3ZDLEtBQStCLENBQUM7UUFFcEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxTQUFTLENBQUMsaUJBQWlCLENBQUMsU0FBUyxDQUFDLEVBQ3RDLEdBQUcsRUFBRSxDQUFDLDZEQUE2RDtZQUMvRCxzREFBc0QsU0FBUyxHQUFHLENBQUMsQ0FBQztRQUU1RSxPQUFPO1lBQ0wsQ0FBQyxFQUFFLEdBQUcsRUFBRSxDQUNKLG1CQUFtQixDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsRUFBRSxFQUFFLE9BQU8sRUFBRSxPQUFPLEVBQUUsR0FBRyxFQUFFLFVBQVUsQ0FBQztZQUN6RSxNQUFNLEVBQUUsR0FBRyxFQUFFLENBQ1Qsb0JBQW9CLENBQUMsR0FBRyxFQUFFLEVBQUUsRUFBRSxPQUFPLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxHQUFHLEVBQUUsVUFBVSxDQUFDO1NBQzNFLENBQUM7SUFDSixDQUFDO0NBQ0YsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7Q29udjJELCBDb252MkRBdHRyc30gZnJvbSAnLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7R3JhZENvbmZpZywgTmFtZWRBdHRyTWFwfSBmcm9tICcuLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtjb252MkRCYWNrcHJvcEZpbHRlcn0gZnJvbSAnLi4vb3BzL2NvbnYyZF9iYWNrcHJvcF9maWx0ZXInO1xuaW1wb3J0IHtjb252MkRCYWNrcHJvcElucHV0fSBmcm9tICcuLi9vcHMvY29udjJkX2JhY2twcm9wX2lucHV0JztcbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi9vcHMvY29udl91dGlsJztcbmltcG9ydCB7VGVuc29yLCBUZW5zb3I0RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmV4cG9ydCBjb25zdCBjb252MkRHcmFkQ29uZmlnOiBHcmFkQ29uZmlnID0ge1xuICBrZXJuZWxOYW1lOiBDb252MkQsXG4gIGlucHV0c1RvU2F2ZTogWyd4JywgJ2ZpbHRlciddLFxuICBncmFkRnVuYzogKGR5OiBUZW5zb3I0RCwgc2F2ZWQ6IFRlbnNvcltdLCBhdHRyczogTmFtZWRBdHRyTWFwKSA9PiB7XG4gICAgY29uc3QgW3g0RCwgJGZpbHRlcl0gPSBzYXZlZCBhcyBbVGVuc29yNEQsIFRlbnNvcjREXTtcbiAgICBjb25zdCB7ZGlsYXRpb25zLCBzdHJpZGVzLCBwYWQsIGRhdGFGb3JtYXR9ID1cbiAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBDb252MkRBdHRycztcblxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjb252X3V0aWwudHVwbGVWYWx1ZXNBcmVPbmUoZGlsYXRpb25zKSxcbiAgICAgICAgKCkgPT4gJ0Vycm9yIGluIGdyYWRpZW50IG9mIGNvbnYyRDogZGlsYXRpb24gcmF0ZXMgZ3JlYXRlciB0aGFuIDEgJyArXG4gICAgICAgICAgICBgYXJlIG5vdCB5ZXQgc3VwcG9ydGVkIGluIGdyYWRpZW50cy4gR290IGRpbGF0aW9ucyAnJHtkaWxhdGlvbnN9J2ApO1xuXG4gICAgcmV0dXJuIHtcbiAgICAgIHg6ICgpID0+XG4gICAgICAgICAgY29udjJEQmFja3Byb3BJbnB1dCh4NEQuc2hhcGUsIGR5LCAkZmlsdGVyLCBzdHJpZGVzLCBwYWQsIGRhdGFGb3JtYXQpLFxuICAgICAgZmlsdGVyOiAoKSA9PlxuICAgICAgICAgIGNvbnYyREJhY2twcm9wRmlsdGVyKHg0RCwgZHksICRmaWx0ZXIuc2hhcGUsIHN0cmlkZXMsIHBhZCwgZGF0YUZvcm1hdClcbiAgICB9O1xuICB9XG59O1xuIl19