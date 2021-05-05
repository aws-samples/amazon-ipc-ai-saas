#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from '@aws-cdk/core';
import { IpcAiSaasStack } from '../lib/ipc-ai-saas-stack';
import { BootstraplessStackSynthesizer } from 'cdk-bootstrapless-synthesizer';

const app = new cdk.App();
new IpcAiSaasStack(app, 'IpcAiSaasStack', { synthesizer: newSynthesizer() });

app.synth();

function newSynthesizer() {
  return process.env.USE_BSS ? new BootstraplessStackSynthesizer(): undefined;
}