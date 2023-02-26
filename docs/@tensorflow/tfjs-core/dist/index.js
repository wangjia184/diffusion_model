/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
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
// Required side effectful code.
import './base_side_effects';
// TODO(mattSoulanille): Move this to base_side_effects.ts
// It is here for now because custom bundles need to avoid calling it, and they
// only replace the index.js file, not the base_side_effects file.
import { registerOptimizers } from './optimizers/register_optimizers';
registerOptimizers();
// All exports from this package should be in base.
export * from './base';
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiaW5kZXguanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL2luZGV4LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILGdDQUFnQztBQUNoQyxPQUFPLHFCQUFxQixDQUFDO0FBRTdCLDBEQUEwRDtBQUMxRCwrRUFBK0U7QUFDL0Usa0VBQWtFO0FBQ2xFLE9BQU8sRUFBQyxrQkFBa0IsRUFBQyxNQUFNLGtDQUFrQyxDQUFDO0FBQ3BFLGtCQUFrQixFQUFFLENBQUM7QUFFckIsbURBQW1EO0FBQ25ELGNBQWMsUUFBUSxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vLyBSZXF1aXJlZCBzaWRlIGVmZmVjdGZ1bCBjb2RlLlxuaW1wb3J0ICcuL2Jhc2Vfc2lkZV9lZmZlY3RzJztcblxuLy8gVE9ETyhtYXR0U291bGFuaWxsZSk6IE1vdmUgdGhpcyB0byBiYXNlX3NpZGVfZWZmZWN0cy50c1xuLy8gSXQgaXMgaGVyZSBmb3Igbm93IGJlY2F1c2UgY3VzdG9tIGJ1bmRsZXMgbmVlZCB0byBhdm9pZCBjYWxsaW5nIGl0LCBhbmQgdGhleVxuLy8gb25seSByZXBsYWNlIHRoZSBpbmRleC5qcyBmaWxlLCBub3QgdGhlIGJhc2Vfc2lkZV9lZmZlY3RzIGZpbGUuXG5pbXBvcnQge3JlZ2lzdGVyT3B0aW1pemVyc30gZnJvbSAnLi9vcHRpbWl6ZXJzL3JlZ2lzdGVyX29wdGltaXplcnMnO1xucmVnaXN0ZXJPcHRpbWl6ZXJzKCk7XG5cbi8vIEFsbCBleHBvcnRzIGZyb20gdGhpcyBwYWNrYWdlIHNob3VsZCBiZSBpbiBiYXNlLlxuZXhwb3J0ICogZnJvbSAnLi9iYXNlJztcbiJdfQ==