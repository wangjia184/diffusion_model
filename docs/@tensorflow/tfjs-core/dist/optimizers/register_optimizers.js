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
import { AdadeltaOptimizer } from './adadelta_optimizer';
import { AdagradOptimizer } from './adagrad_optimizer';
import { AdamOptimizer } from './adam_optimizer';
import { AdamaxOptimizer } from './adamax_optimizer';
import { MomentumOptimizer } from './momentum_optimizer';
import { RMSPropOptimizer } from './rmsprop_optimizer';
import { SGDOptimizer } from './sgd_optimizer';
import { registerClass } from '../serialization';
const OPTIMIZERS = [
    AdadeltaOptimizer,
    AdagradOptimizer,
    AdamOptimizer,
    AdamaxOptimizer,
    MomentumOptimizer,
    RMSPropOptimizer,
    SGDOptimizer,
];
export function registerOptimizers() {
    for (const optimizer of OPTIMIZERS) {
        registerClass(optimizer);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVnaXN0ZXJfb3B0aW1pemVycy5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3B0aW1pemVycy9yZWdpc3Rlcl9vcHRpbWl6ZXJzLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxpQkFBaUIsRUFBQyxNQUFNLHNCQUFzQixDQUFDO0FBQ3ZELE9BQU8sRUFBQyxnQkFBZ0IsRUFBQyxNQUFNLHFCQUFxQixDQUFDO0FBQ3JELE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUMvQyxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sb0JBQW9CLENBQUM7QUFDbkQsT0FBTyxFQUFDLGlCQUFpQixFQUFDLE1BQU0sc0JBQXNCLENBQUM7QUFDdkQsT0FBTyxFQUFDLGdCQUFnQixFQUFDLE1BQU0scUJBQXFCLENBQUM7QUFDckQsT0FBTyxFQUFDLFlBQVksRUFBQyxNQUFNLGlCQUFpQixDQUFDO0FBQzdDLE9BQU8sRUFBQyxhQUFhLEVBQUMsTUFBTSxrQkFBa0IsQ0FBQztBQUUvQyxNQUFNLFVBQVUsR0FBRztJQUNqQixpQkFBaUI7SUFDakIsZ0JBQWdCO0lBQ2hCLGFBQWE7SUFDYixlQUFlO0lBQ2YsaUJBQWlCO0lBQ2pCLGdCQUFnQjtJQUNoQixZQUFZO0NBQ2IsQ0FBQztBQUVGLE1BQU0sVUFBVSxrQkFBa0I7SUFDaEMsS0FBSyxNQUFNLFNBQVMsSUFBSSxVQUFVLEVBQUU7UUFDbEMsYUFBYSxDQUFDLFNBQVMsQ0FBQyxDQUFDO0tBQzFCO0FBQ0gsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIyIEdvb2dsZSBMTEMuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtBZGFkZWx0YU9wdGltaXplcn0gZnJvbSAnLi9hZGFkZWx0YV9vcHRpbWl6ZXInO1xuaW1wb3J0IHtBZGFncmFkT3B0aW1pemVyfSBmcm9tICcuL2FkYWdyYWRfb3B0aW1pemVyJztcbmltcG9ydCB7QWRhbU9wdGltaXplcn0gZnJvbSAnLi9hZGFtX29wdGltaXplcic7XG5pbXBvcnQge0FkYW1heE9wdGltaXplcn0gZnJvbSAnLi9hZGFtYXhfb3B0aW1pemVyJztcbmltcG9ydCB7TW9tZW50dW1PcHRpbWl6ZXJ9IGZyb20gJy4vbW9tZW50dW1fb3B0aW1pemVyJztcbmltcG9ydCB7Uk1TUHJvcE9wdGltaXplcn0gZnJvbSAnLi9ybXNwcm9wX29wdGltaXplcic7XG5pbXBvcnQge1NHRE9wdGltaXplcn0gZnJvbSAnLi9zZ2Rfb3B0aW1pemVyJztcbmltcG9ydCB7cmVnaXN0ZXJDbGFzc30gZnJvbSAnLi4vc2VyaWFsaXphdGlvbic7XG5cbmNvbnN0IE9QVElNSVpFUlMgPSBbXG4gIEFkYWRlbHRhT3B0aW1pemVyLFxuICBBZGFncmFkT3B0aW1pemVyLFxuICBBZGFtT3B0aW1pemVyLFxuICBBZGFtYXhPcHRpbWl6ZXIsXG4gIE1vbWVudHVtT3B0aW1pemVyLFxuICBSTVNQcm9wT3B0aW1pemVyLFxuICBTR0RPcHRpbWl6ZXIsXG5dO1xuXG5leHBvcnQgZnVuY3Rpb24gcmVnaXN0ZXJPcHRpbWl6ZXJzKCkge1xuICBmb3IgKGNvbnN0IG9wdGltaXplciBvZiBPUFRJTUlaRVJTKSB7XG4gICAgcmVnaXN0ZXJDbGFzcyhvcHRpbWl6ZXIpO1xuICB9XG59XG4iXX0=