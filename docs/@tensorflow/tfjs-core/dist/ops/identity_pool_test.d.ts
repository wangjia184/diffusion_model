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
/// <amd-module name="@tensorflow/tfjs-core/dist/ops/identity_pool_test" />
import * as tf from '../index';
/**
 * Test utility for testing AvgPool, MaxPool, etc where kernel size is 1x1,
 * effectively making them act as the identity function except where strides
 * affect the output.
 */
export declare function identityPoolTest(pool: typeof tf.avgPool): void;
