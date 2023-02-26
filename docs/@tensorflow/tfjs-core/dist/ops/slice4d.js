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
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
import { slice } from './slice';
/**
 * Extracts a 4D slice from a 4D array starting at coordinates `begin` and
 * is of size `size`. See `slice` for details.
 */
function slice4d_(x, begin, size) {
    const $x = convertToTensor(x, 'x', 'slice4d');
    util.assert($x.rank === 4, () => `slice4d expects a rank-4 tensor, but got a rank-${$x.rank} tensor`);
    return slice($x, begin, size);
}
export const slice4d = /* @__PURE__ */ op({ slice4d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2xpY2U0ZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NsaWNlNGQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLEtBQUssRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUU5Qjs7O0dBR0c7QUFDSCxTQUFTLFFBQVEsQ0FDYixDQUFzQixFQUFFLEtBQXVDLEVBQy9ELElBQXNDO0lBQ3hDLE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQzlDLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2IsR0FBRyxFQUFFLENBQ0QsbURBQW1ELEVBQUUsQ0FBQyxJQUFJLFNBQVMsQ0FBQyxDQUFDO0lBQzdFLE9BQU8sS0FBSyxDQUFDLEVBQUUsRUFBRSxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUM7QUFDaEMsQ0FBQztBQUNELE1BQU0sQ0FBQyxNQUFNLE9BQU8sR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsUUFBUSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3I0RH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3NsaWNlfSBmcm9tICcuL3NsaWNlJztcblxuLyoqXG4gKiBFeHRyYWN0cyBhIDREIHNsaWNlIGZyb20gYSA0RCBhcnJheSBzdGFydGluZyBhdCBjb29yZGluYXRlcyBgYmVnaW5gIGFuZFxuICogaXMgb2Ygc2l6ZSBgc2l6ZWAuIFNlZSBgc2xpY2VgIGZvciBkZXRhaWxzLlxuICovXG5mdW5jdGlvbiBzbGljZTRkXyhcbiAgICB4OiBUZW5zb3I0RHxUZW5zb3JMaWtlLCBiZWdpbjogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgc2l6ZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pOiBUZW5zb3I0RCB7XG4gIGNvbnN0ICR4ID0gY29udmVydFRvVGVuc29yKHgsICd4JywgJ3NsaWNlNGQnKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkeC5yYW5rID09PSA0LFxuICAgICAgKCkgPT5cbiAgICAgICAgICBgc2xpY2U0ZCBleHBlY3RzIGEgcmFuay00IHRlbnNvciwgYnV0IGdvdCBhIHJhbmstJHskeC5yYW5rfSB0ZW5zb3JgKTtcbiAgcmV0dXJuIHNsaWNlKCR4LCBiZWdpbiwgc2l6ZSk7XG59XG5leHBvcnQgY29uc3Qgc2xpY2U0ZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7c2xpY2U0ZF99KTtcbiJdfQ==